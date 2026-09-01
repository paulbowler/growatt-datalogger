"""Shared plumbing for the entities that write registers.

Write entities differ from sensors in where their value comes from. A telemetry record
carries input registers; these settings live in the holding space. Fortunately an
announce carries holding registers, so for most of them the device volunteers the current
value every time it connects, and the entity simply reads it from the coordinator.

For a register no announce reports, the entity asks for it directly -- but only once a
record has arrived, because entities are added during setup, before any datalogger has
connected. Reading at add time talks to nothing, and as a one-shot it would never retry,
leaving the entity unknown for good.

A write that the device rejects does not update the state. Optimistic updates would be
worse than useless here: showing a battery cut-off the inverter never accepted is exactly
the sort of thing someone builds an automation on.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from growatt_protocol import CommandTimeout, commands
from growatt_protocol.registers.writable import (
    BY_KEY,
    Encoding,
    WritableRegister,
    WriteKind,
    for_profile,
)
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import DOMAIN, KIND_INVERTER, SIGNAL_NEW_DEVICE
from .entity import GrowattEntity
from .hub import GrowattDevice, GrowattHub
from .metadata import pretty

_LOGGER = logging.getLogger(__name__)

_SCHEDULE_ENABLE_REGISTERS = {
    1082,
    1085,
    1088,
    1102,
    1105,
    1108,
}
_SCHEDULE_ENABLE_SLOTS = {
    "grid_first_enabled_1": ("grid_first_start_time_1", "grid_first_stop_time_1"),
    "grid_first_enabled_2": ("grid_first_start_time_2", "grid_first_stop_time_2"),
    "grid_first_enabled_3": ("grid_first_start_time_3", "grid_first_stop_time_3"),
    "battery_first_enabled_1": ("battery_first_start_time", "battery_first_stop_time"),
    "battery_first_enabled_2": ("battery_first_start_time_2", "battery_first_stop_time_2"),
    "battery_first_enabled_3": ("battery_first_start_time_3", "battery_first_stop_time_3"),
}
_DOMAIN_BY_KIND = {
    WriteKind.NUMBER: "number",
    WriteKind.SELECT: "select",
    WriteKind.SWITCH: "switch",
    WriteKind.TIME: "time",
}
_UNKNOWN_STATES = {"", "unknown", "unavailable"}

_READ_GROUPS = (
    (1070, 1071),
    (1080, 1088),
    (1090, 1092),
    (1100, 1108),
)
_READ_GROUP_TASKS: dict[
    tuple[int, str, int, int], asyncio.Task[dict[int, int]]
] = {}


def async_setup_write_platform(
    hass: HomeAssistant,
    entry: Any,
    async_add_entities: AddConfigEntryEntitiesCallback,
    kind: WriteKind,
    factory: Callable[[GrowattHub, GrowattDevice, WritableRegister], GrowattEntity],
) -> None:
    """Create write entities of one kind as inverters are discovered."""
    hub: GrowattHub = entry.runtime_data
    created: set[tuple[str, str]] = set()

    @callback
    def _add(device_key: str, _names: list[str]) -> None:
        device = hub.devices.get(device_key)
        if device is None or device.kind != KIND_INVERTER or device.profile is None:
            return

        entities = []
        for spec in for_profile(device.profile, include_unverified=True):
            if spec.kind is not kind:
                continue
            token = (device_key, spec.key)
            if token in created:
                continue
            created.add(token)
            entities.append(factory(hub, device, spec))

        if entities:
            async_add_entities(entities)

    entry.async_on_unload(
        async_dispatcher_connect(hass, SIGNAL_NEW_DEVICE.format(entry_id=entry.entry_id), _add)
    )
    hub.async_replay(_add)


class GrowattWriteEntity(GrowattEntity):
    """Base for an entity backed by a writable holding register."""

    _attr_entity_category = EntityCategory.CONFIG

    def __init__(self, hub: GrowattHub, device: GrowattDevice, spec: WritableRegister) -> None:
        super().__init__(hub, device, spec.key)
        self.spec = spec
        self._attr_name = pretty(spec.key)
        self._attr_icon = spec.icon
        self._attr_entity_registry_enabled_default = spec.enabled_default
        self._current: Any = None
        self._refresh_requested = False
        self._refresh_gave_empty_response = False

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        # Surfacing the provenance means a user can judge for themselves whether to
        # trust a register this project has flagged as unverified.
        return {
            "register": self.spec.register,
            "confidence": self.spec.confidence.value,
            "source": self.spec.source,
        }

    @property
    def _reported(self) -> Any | None:
        """This register's value as the device itself last reported it.

        An announce carries holding registers, which is where these settings live, so
        for most of them the device volunteers the current value every time it connects
        -- no command round-trip needed, and it refreshes itself.

        Only unscaled encodings are taken this way. A scaled one would already have been
        divided by the register table, and running it through :meth:`decode` again would
        scale it twice.
        """
        if self.spec.encoding not in (Encoding.RAW, Encoding.BOOL):
            return None
        value = (self.coordinator.data or {}).get(self.spec.key)
        if not isinstance(value, int):
            return None
        return self.spec.decode(value)

    @property
    def _state(self) -> Any | None:
        """What to display: the device's own report, else our last read or write."""
        reported = self._reported
        return self._current if reported is None else reported

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        # No read here. Entities are added during setup, before any datalogger has
        # connected, so a read at this point has nothing to talk to and -- being a
        # one-shot -- would never be retried, leaving the entity unknown for good. The
        # value comes from the announce instead, and _handle_coordinator_update asks
        # explicitly only for the registers an announce does not carry.
        self._refresh_requested = False

    @callback
    def _handle_coordinator_update(self) -> None:
        # A record has arrived, so the device is connected and a command can be sent.
        if (
            not self._refresh_requested
            and not self._refresh_gave_empty_response
            and self._reported is None
            and self._current is None
        ):
            self._refresh_requested = True
            self.hass.async_create_background_task(
                self._async_refresh(),
                name=f"growatt read {self.device.serial} {self.spec.key}",
            )
        super()._handle_coordinator_update()

    async def _async_refresh(self) -> None:
        """Read the register back. Leaves the value unknown if it cannot be read."""
        session = self._session()
        if session is None:
            self._refresh_requested = False
            return
        try:
            word = await self._async_read_register_word(session)
        except (CommandTimeout, ConnectionError) as err:
            _LOGGER.debug("could not read %s: %s", self.spec.key, err)
            return
        finally:
            self._refresh_requested = False

        if word is None:
            # The device does not implement this register. Better an unknown value than
            # a plausible-looking wrong one.
            _LOGGER.debug(
                "%s does not implement register %s", self.device.serial, self.spec.register
            )
            self._refresh_gave_empty_response = True
            return

        self._current = self.spec.decode(word)
        self.async_write_ha_state()

    async def _async_read_register_word(self, session: Any) -> int | None:
        if group := _read_group_for(self.spec.register):
            values = await _async_read_group_cached(session, *group)
            if self.spec.register in values:
                return values[self.spec.register]

        response = await session.send_command(
            commands.read_inverter(
                session.datalogger_serial, session.protocol, self.spec.register
            )
        )
        if response.empty or response.value is None:
            return None
        return int(response.value)

    async def _async_write(self, value: Any) -> None:
        """Write ``value``, then read the register back to confirm."""
        session = self._session()
        if session is None:
            raise HomeAssistantError(f"Datalogger for {self.device.serial} is not connected")

        try:
            word = self.spec.encode(value)
        except ValueError as err:
            raise HomeAssistantError(str(err)) from err

        try:
            if self.spec.register in _SCHEDULE_ENABLE_REGISTERS:
                response = await self._async_write_schedule_enable(session, word)
            else:
                response = await session.send_command(
                    commands.write_inverter(
                        session.datalogger_serial, session.protocol, self.spec.register, word
                    )
                )
        except (CommandTimeout, ConnectionError) as err:
            # Deliberately not retried: repeating a write could apply a change twice.
            raise HomeAssistantError(
                f"{self.spec.key} was not confirmed: {err}. Reload or read the register "
                "back to see whether it took effect."
            ) from err

        if not response.ok:
            raise HomeAssistantError(
                f"The inverter rejected {self.spec.key} with result {response.result}"
            )

        self._refresh_gave_empty_response = False
        await self._async_refresh()

    async def _async_write_schedule_enable(self, session: Any, word: int) -> Any:
        """Write an enable flag together with its start/stop time registers.

        Some SPH firmware rejects changing a schedule slot's enable flag as a lone
        0x06 write. Preserving the adjacent start/stop words and applying the complete
        3-register slot as one 0x10 write mirrors how the inverter stores these slots.
        """
        start = self.spec.register - 2
        values: list[int] | None = None
        try:
            read = await session.send_command(
                commands.read_inverter(
                    session.datalogger_serial, session.protocol, start, self.spec.register
                )
            )
            if not read.empty and len(read.values) >= 3:
                values = list(read.values[:3])
        except (CommandTimeout, ConnectionError) as err:
            _LOGGER.debug("could not pre-read schedule slot for %s: %s", self.spec.key, err)

        if values is None:
            values = self._schedule_slot_words_from_state(word)
        if values is None:
            raise HomeAssistantError(
                f"Could not confirm schedule slot {start}-{self.spec.register} before writing "
                f"{self.spec.key}; wait for the start/stop time entities to populate and try again"
            )

        values[2] = word
        return await session.send_command(
            commands.write_inverter_range(session.datalogger_serial, session.protocol, start, values)
        )

    def _schedule_slot_words_from_state(self, word: int) -> list[int] | None:
        keys = _SCHEDULE_ENABLE_SLOTS.get(self.spec.key)
        if keys is None:
            return None

        encoded: list[int] = []
        for key in keys:
            state = self._state_for_writable_key(key)
            if state is None:
                return None
            try:
                encoded.append(BY_KEY[key].encode(state))
            except (KeyError, ValueError):
                return None
        encoded.append(word)
        return encoded

    def _state_for_writable_key(self, key: str) -> str | None:
        spec = BY_KEY.get(key)
        if spec is None:
            return None
        domain = _DOMAIN_BY_KIND[spec.kind]
        unique_id = f"{DOMAIN}_{self.device.key}_{key}"
        entity_id = er.async_get(self.hass).async_get_entity_id(domain, DOMAIN, unique_id)
        if entity_id is None:
            return None
        state = self.hass.states.get(entity_id)
        if state is None or state.state in _UNKNOWN_STATES:
            return None
        return state.state

    def _session(self) -> Any:
        parent = self.device.parent
        if parent is None:
            return None
        return self.hub.session_for(self.hub.devices[parent].serial)


def _read_group_for(register: int) -> tuple[int, int] | None:
    for start, end in _READ_GROUPS:
        if start <= register <= end:
            return start, end
    return None


async def _async_read_group_cached(session: Any, start: int, end: int) -> dict[int, int]:
    key = (id(session), session.datalogger_serial, start, end)
    task = _READ_GROUP_TASKS.get(key)
    if task is None or task.done():
        task = asyncio.create_task(_async_read_group(session, start, end))
        _READ_GROUP_TASKS[key] = task

    try:
        return await task
    finally:
        if task.done() and _READ_GROUP_TASKS.get(key) is task:
            _READ_GROUP_TASKS.pop(key, None)


async def _async_read_group(session: Any, start: int, end: int) -> dict[int, int]:
    response = await session.send_command(
        commands.read_inverter(session.datalogger_serial, session.protocol, start, end)
    )
    if response.empty:
        return {}
    return {start + index: value for index, value in enumerate(response.values)}
