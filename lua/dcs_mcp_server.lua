--[[
  dcs_mcp_server.lua
  Project Ender — DCS mission socket server

  Listens on TCP port 7374 for newline-delimited JSON commands from the MCP server
  running on the Linux host. Non-blocking; polled every 0.1s via timer.scheduleFunction.

  Protocol:
    Request:  {"cmd": "get_state"}\n
    Request:  {"cmd": "spawn_flight", "airport": "Batumi", "type": "FA-18C_hornet",
                "count": 2, "callsign": "Enfield", "coalition": "blue"}\n
    Response: {"ok": true, "data": {...}}\n  |  {"ok": false, "error": "..."}\n

  MOOSE is loaded before this script via a separate trigger. Its presence is optional
  for the features implemented here — we use the vanilla DCS scripting API for
  reliability, but MOOSE is available for future extension.
--]]

local ENDER_PORT = 7374
local POLL_INTERVAL = 0.1   -- seconds between polls
local MAX_CLIENTS = 4
local READ_CHUNK = 4096

-- ---------------------------------------------------------------------------
-- JSON codec
-- DCS bundles a small cjson library; fall back to a trivial encoder if absent.
-- ---------------------------------------------------------------------------
local json = {}

if pcall(function() json = require("cjson") end) then
    -- cjson available — use it
elseif pcall(function() json = require("json") end) then
    -- alternate binding name
else
    -- Minimal fallback encoder/decoder (handles flat tables with string/number/bool values)
    local function encode_value(v)
        local t = type(v)
        if t == "string" then
            return '"' .. v:gsub('\\', '\\\\'):gsub('"', '\\"'):gsub('\n', '\\n') .. '"'
        elseif t == "number" or t == "boolean" then
            return tostring(v)
        elseif t == "nil" then
            return "null"
        elseif t == "table" then
            -- array check: keys are 1..n
            local is_array = true
            local n = 0
            for k, _ in pairs(v) do
                n = n + 1
                if type(k) ~= "number" or k ~= math.floor(k) then
                    is_array = false
                    break
                end
            end
            if is_array and n > 0 then
                local parts = {}
                for i = 1, n do parts[i] = encode_value(v[i]) end
                return "[" .. table.concat(parts, ",") .. "]"
            else
                local parts = {}
                for k, val in pairs(v) do
                    table.insert(parts, '"' .. tostring(k) .. '":' .. encode_value(val))
                end
                return "{" .. table.concat(parts, ",") .. "}"
            end
        end
        return "null"
    end

    json.encode = encode_value

    -- Minimal decoder: only handles the command objects we expect.
    -- Extracts string fields by pattern matching — good enough for controlled input.
    json.decode = function(s)
        local result = {}
        -- Extract string values: "key": "value"
        for k, v in s:gmatch('"(%w+)"%s*:%s*"([^"]*)"') do
            result[k] = v
        end
        -- Extract numeric values: "key": number
        for k, v in s:gmatch('"(%w+)"%s*:%s*(-?%d+%.?%d*)') do
            if result[k] == nil then result[k] = tonumber(v) end
        end
        return result
    end
end

-- ---------------------------------------------------------------------------
-- Helpers
-- ---------------------------------------------------------------------------

local function ok_response(data)
    return json.encode({ ok = true, data = data }) .. "\n"
end

local function err_response(msg)
    return json.encode({ ok = false, error = msg }) .. "\n"
end

local function coalition_name(side)
    if side == coalition.side.BLUE then return "blue"
    elseif side == coalition.side.RED then return "red"
    else return "neutral"
    end
end

local function coalition_side(name)
    if name == "blue" then return coalition.side.BLUE
    elseif name == "red" then return coalition.side.RED
    else return coalition.side.NEUTRAL
    end
end

local function country_id_for_coalition(side)
    -- Returns a representative country ID for addGroup calls.
    -- Blue → USA (2), Red → Russia (0) in Caucasus.
    if side == coalition.side.BLUE then return country.id.USA
    elseif side == coalition.side.RED then return country.id.RUSSIA
    else return country.id.RUSSIA
    end
end

-- ---------------------------------------------------------------------------
-- Command handlers
-- ---------------------------------------------------------------------------

local function handle_get_state()
    local units = {}

    for _, side in ipairs({ coalition.side.BLUE, coalition.side.RED, coalition.side.NEUTRAL }) do
        local groups = coalition.getGroups(side, Group.Category.AIRPLANE)
        for _, grp in ipairs(groups or {}) do
            for _, unit in ipairs(grp:getUnits() or {}) do
                if unit:isExist() then
                    local pos = unit:getPoint()
                    local ll = coord.LOtoLL(pos)
                    local heading = math.deg(unit:getHeading())
                    table.insert(units, {
                        name      = unit:getName(),
                        type      = unit:getTypeName(),
                        coalition = coalition_name(side),
                        lat       = ll.Lat,
                        lon       = ll.Long,
                        alt       = math.floor(pos.y),
                        heading   = math.floor(heading),
                        alive     = unit:getLife() > 0,
                    })
                end
            end
        end
    end

    local airbases = {}
    for _, ab in ipairs(world.getAirbases() or {}) do
        table.insert(airbases, {
            name      = ab:getName(),
            coalition = coalition_name(ab:getCoalition()),
        })
    end

    return ok_response({
        time     = timer.getTime(),
        units    = units,
        airbases = airbases,
    })
end

-- Spawn counter ensures unique group names within a session.
local _spawn_counter = 0

local function handle_spawn_flight(cmd)
    local airport_name = cmd.airport
    local ac_type      = cmd.type or "FA-18C_hornet"
    local count        = tonumber(cmd.count) or 2
    local callsign     = cmd.callsign or "Ender"
    local side         = coalition_side(cmd.coalition or "blue")

    if not airport_name then
        return err_response("missing field: airport")
    end

    -- Find the airbase and get its position.
    local ab = Airbase.getByName(airport_name)
    if not ab then
        return err_response("airbase not found: " .. airport_name)
    end

    local ab_pos = ab:getPoint()

    _spawn_counter = _spawn_counter + 1
    local group_name = callsign .. "-" .. _spawn_counter

    -- Build unit table for coalition.addGroup.
    -- Units are placed in a line 50 m apart on the ramp; DCS will park them properly.
    local group_units = {}
    for i = 1, count do
        table.insert(group_units, {
            ["type"]         = ac_type,
            ["name"]         = group_name .. " " .. i,
            ["unitId"]       = 10000 + _spawn_counter * 10 + i,
            ["skill"]        = "High",
            ["playerCanDrive"] = false,
            ["x"]            = ab_pos.x + (i - 1) * 50,
            ["y"]            = ab_pos.z + (i - 1) * 10,   -- DCS z == map north
            ["alt"]          = 0,
            ["alt_type"]     = "BARO",
            ["speed"]        = 0,
            ["heading"]      = 0,
            ["payload"]      = { ["pylons"] = {}, ["fuel"] = 5030, ["flare"] = 30, ["chaff"] = 30, ["gun"] = 100 },
            ["callsign"]     = { [1] = 1, [2] = 1, [3] = i, ["name"] = callsign .. "1" .. i },
        })
    end

    local group_data = {
        ["groupId"]   = 10000 + _spawn_counter,
        ["name"]      = group_name,
        ["task"]      = "Nothing",
        ["start_time"] = 0,
        ["tasks"]     = {},
        ["route"]     = { ["points"] = {
            { ["x"] = ab_pos.x, ["y"] = ab_pos.z, ["alt"] = 0, ["alt_type"] = "BARO",
              ["speed"] = 0, ["action"] = "From Parking Area Hot", ["type"] = "TakeOffParking",
              ["airdromeId"] = ab:getID(), ["ETA"] = 0, ["ETA_locked"] = true, ["name"] = "WP1" }
        }},
        ["units"]     = group_units,
        ["hidden"]    = false,
        ["uncontrolled"] = true,   -- parked / cold
    }

    local country_id = country_id_for_coalition(side)
    local grp = coalition.addGroup(country_id, Group.Category.AIRPLANE, group_data)

    if grp then
        return ok_response({ group_name = group_name, unit_count = count })
    else
        return err_response("coalition.addGroup failed — check DCS.log for details")
    end
end

local function dispatch(line)
    local ok, cmd = pcall(json.decode, line)
    if not ok or type(cmd) ~= "table" then
        return err_response("json parse error")
    end

    if cmd.cmd == "get_state" then
        return handle_get_state()
    elseif cmd.cmd == "spawn_flight" then
        return handle_spawn_flight(cmd)
    else
        return err_response("unknown command: " .. tostring(cmd.cmd))
    end
end

-- ---------------------------------------------------------------------------
-- Server lifecycle
-- ---------------------------------------------------------------------------

local socket = require("socket")

local server = socket.tcp()
server:setoption("reuseaddr", true)
local bound, bind_err = server:bind("0.0.0.0", ENDER_PORT)
if not bound then
    env.warning("[Ender] Failed to bind port " .. ENDER_PORT .. ": " .. tostring(bind_err))
    return
end
server:listen(MAX_CLIENTS)
server:settimeout(0)   -- non-blocking accept

-- Per-client state: {socket, buffer}
local clients = {}

local function poll(_, t)
    -- Accept new connections
    local client = server:accept()
    if client then
        client:settimeout(0)
        table.insert(clients, { sock = client, buf = "" })
        env.info("[Ender] client connected (" .. #clients .. " total)")
    end

    -- Service existing clients
    local to_remove = {}
    for i, c in ipairs(clients) do
        local chunk, err = c.sock:receive(READ_CHUNK)
        if chunk then
            c.buf = c.buf .. chunk
            -- Process all complete lines in the buffer
            while true do
                local nl = c.buf:find("\n", 1, true)
                if not nl then break end
                local line = c.buf:sub(1, nl - 1)
                c.buf = c.buf:sub(nl + 1)
                local response = dispatch(line)
                c.sock:send(response)
            end
        elseif err == "closed" then
            env.info("[Ender] client disconnected")
            c.sock:close()
            table.insert(to_remove, i)
        end
        -- "timeout" (no data yet) is normal — skip
    end

    -- Remove closed clients (reverse order to preserve indices)
    for i = #to_remove, 1, -1 do
        table.remove(clients, to_remove[i])
    end

    return t + POLL_INTERVAL
end

-- Start polling 1 second after mission load.
timer.scheduleFunction(poll, nil, timer.getTime() + 1)
env.info("[Ender] DCS MCP socket server listening on :" .. ENDER_PORT)
