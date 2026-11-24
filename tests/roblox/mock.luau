-- A small model of the parts of Roblox the backend uses, for tests that run in Lute: an instance tree with parents and
-- children, ModuleScripts and Scripts that run from their source with their own `script` and `require`, Actors that each
-- own a VM (their own module cache and globals), `SendMessage` and `BindToMessage` with copied arguments and a
-- delivery latency in frames, `task`, and an `HttpService` that serves what a test gives it. It also enforces the rules
-- of docs/roblox.md that decide whether code works at all: a script over 199,997 characters arrives empty, and a module
-- must return one value, `BindToMessage` only from a script under the Actor, a message
-- sent before its topic is bound is dropped, a script starts a frame after it is parented, and `require`, `RequestAsync`
-- and `SetAttribute` are refused in the parallel phase, which a thread enters with `task.desynchronize` or when a
-- parallel handler wakes it. It checks that the backend is wired right. It does not say how Roblox behaves: message
-- latency, timing and limits are assumptions written from the documentation.
const luau = require("@lute/luau")

export type Instance = any

export type Mock = {
    --- The root of the instance tree, which is `game`.
    game: Instance,
    --- A new instance of a class, `Instance.new`.
    new: (className: string) -> Instance,
    --- Runs `main` as the code of the script `owner` (none when nil) and the scheduler until it returns. Errors when it
    --- is stuck waiting.
    run: (main: (...any) -> ...any, owner: Instance?) -> (),
    --- The phase of the running thread, "serial" or "parallel".
    phase: () -> string,
    --- Messages that arrived at a topic nothing was bound to.
    dropped: () -> number,
    --- The `require` of the main VM.
    require: (module: Instance) -> any,
    --- The globals the scripts see as `task`.
    task: { [string]: any },
    --- The frames the scheduler has advanced, one per `task.wait` that had nothing else to run, and per message latency.
    frame: () -> number,
    --- Frames between a message sent and delivered, 0 for the same frame.
    setLatency: (frames: number) -> (),
    --- What the HttpService answers, given the request table.
    setHttp: (handler: (request: { [string]: any }) -> { Success: boolean, StatusCode: number, Body: string }) -> (),
    --- Messages sent and bytes of buffers in them since the start.
    messages: () -> (number, number),
    --- The stand ins for `os.clock`, `print` and `warn` the scripts get.
    printed: { string },
}

--- The most characters of source a script holds.
const SOURCE_LIMIT = 199997

type Vm = {
    cache: { [Instance]: any },
    env: { [string]: any },
}

--- A copy of a message argument: buffers and tables are cloned, instances and scalars are shared, as Roblox passes them.
local function copyArgument(v: any, counter: { bytes: number }): any
    if typeof(v) == "buffer" then
        const b = v :: buffer
        const out = buffer.create(buffer.len(b))
        buffer.copy(out, 0, b, 0, buffer.len(b))
        counter.bytes += buffer.len(b)
        return out
    elseif type(v) == "table" and getmetatable(v :: any) == nil then
        const out = {}
        for k, item in v :: any do
            out[k] = copyArgument(item, counter)
        end
        return out
    end
    return v
end

const function new(): Mock
    local frame = 0
    local latency = 0
    local sent = 0
    const counter = { bytes = 0 }
    local httpHandler: ((request: { [string]: any }) -> any)? = nil
    const printed: { string } = {}
    -- Work to resume at a frame: waiting threads, and deliveries.
    const sleeping: { { thread: thread, frame: number } } = {}
    const deliveries: { { frame: number, target: any, topic: string, args: { any } } } = {}
    local task_spawn: (target: any, ...any) -> thread
    -- The threads in the parallel phase, and the script each thread runs for.
    const parallel: { [thread]: boolean } = {}
    const owners: { [thread]: any } = {}
    local dropped = 0

    --- Errors when the running thread is in the parallel phase, where `what` is not allowed.
    const function serialOnly(what: string)
        if parallel[coroutine.running()] then
            error(`{what} is not allowed in the parallel phase`, 0)
        end
    end

    const mock = {} :: Mock
    mock.printed = printed

    --- The class tables of the instance model.
    const methods: { [string]: (self: any, ...any) -> ...any } = {}

    --- The Actor that holds `inst`, itself when it is one, or nil.
    local function actorOf(inst: any): any
        local at = inst
        while at ~= nil do
            if rawget(at, "ClassName") == "Actor" then
                return at
            end
            at = rawget(at, "_parent")
        end
        return nil
    end

    --- Each Actor has a VM, the data model's own is `main`.
    const vms: { [any]: Vm } = {}
    local mainVm: Vm
    local vmOf: (inst: any) -> Vm
    local runScript: (inst: any, vm: Vm) -> any

    const Instance = {}
    local meta: any = {}
    meta.__index = function(self: any, key: string): any
        if key == "Parent" then
            return rawget(self, "_parent")
        end
        const value = rawget(self, key)
        if value ~= nil then
            return value
        end
        const method = methods[key]
        if method ~= nil then
            return method
        end
        for _, child in rawget(self, "_children") do
            if rawget(child, "Name") == key then
                return child
            end
        end
        return nil
    end
    meta.__newindex = function(self: any, key: string, value: any)
        if key == "Parent" then
            const old = rawget(self, "_parent")
            if old ~= nil then
                const list: { any } = (rawget :: any)(old, "_children")
                const at = table.find(list, self)
                if at ~= nil then
                    table.remove(list, at)
                end
            end
            rawset(self, "_parent", value)
            if value ~= nil then
                table.insert((rawget :: any)(value, "_children") :: { any }, self)
                -- A script under an Actor that is in the tree starts when it gets there.
                const function inTree(inst: any): boolean
                    local at = inst
                    while at ~= nil do
                        if at == mock.game then
                            return true
                        end
                        at = rawget(at, "_parent")
                    end
                    return false
                end
                if inTree(self) then
                    const function start(inst: any)
                        if
                            rawget(inst, "ClassName") == "Script"
                            and actorOf(inst) ~= nil
                            and type(rawget(inst, "Source")) == "string"
                        then
                            -- A script starts at the next frame, not when it is parented.
                            const thread = coroutine.create(function()
                                runScript(inst, vmOf(inst))
                            end)
                            owners[thread] = inst
                            table.insert(sleeping, { thread = thread, frame = frame + 1 })
                        end
                        for _, child in table.clone(rawget(inst, "_children")) do
                            start(child)
                        end
                    end
                    start(self)
                end
            end
        elseif
            key == "Source"
            and type(value) == "string"
            and (utf8.len(value :: string) or #(value :: string)) > SOURCE_LIMIT
        then
            -- A script over the limit arrives empty.
            rawset(self, key, "")
        else
            rawset(self, key, value)
        end
    end

    const function create(className: string): any
        const inst: any = setmetatable({ ClassName = className, Name = className, _children = {}, _parent = nil }, meta)
        if className == "Actor" then
            rawset(inst, "_binds", {})
        end
        rawset(inst, "_attrs", {})
        return inst
    end
    function Instance.new(className: string): any
        return create(className)
    end
    mock.new = Instance.new

    function methods.FindFirstChild(self: any, name: string): any
        for _, child in rawget(self, "_children") do
            if rawget(child, "Name") == name then
                return child
            end
        end
        return nil
    end
    function methods.WaitForChild(self: any, name: string): any
        const found = methods.FindFirstChild(self, name)
        if found == nil then
            error(`the mock does not wait for a child: {name} is missing`, 0)
        end
        return found
    end
    function methods.Clone(self: any): any
        const copy = create(rawget(self, "ClassName"))
        for k, v in self :: any do
            if k ~= "_children" and k ~= "_parent" and k ~= "_binds" and k ~= "_attrs" then
                rawset(copy, k, v)
            end
        end
        for _, child in rawget(self, "_children") do
            methods.Clone(child).Parent = copy
        end
        return copy
    end
    function methods.Destroy(self: any)
        serialOnly("Instance:Destroy")
        self.Parent = nil
    end
    function methods.GetActor(self: any): any
        return actorOf(self)
    end
    function methods.GetService(self: any, name: string): any
        local service = methods.FindFirstChild(self, name)
        if service == nil then
            service = create(name)
            rawset(service, "Name", name)
            service.Parent = self
        end
        return service
    end
    function methods.SetAttribute(self: any, name: string, value: any)
        serialOnly("Instance:SetAttribute")
        rawget(self, "_attrs")[name] = value
    end
    function methods.GetAttribute(self: any, name: string): any
        return rawget(self, "_attrs")[name]
    end
    function methods.RequestAsync(_self: any, request: { [string]: any }): any
        serialOnly("HttpService:RequestAsync")
        if httpHandler == nil then
            error("the mock has no HTTP handler", 0)
        end
        return (httpHandler :: any)(request)
    end

    --- Actor messages. A topic is bound by a script under the Actor, in the serial or the parallel phase.
    const function bindTopic(self: any, topic: string, handler: (...any) -> (), inParallel: boolean)
        const owner = owners[coroutine.running()]
        if owner == nil or actorOf(owner) ~= self then
            error("BindToMessage can only be called from a script that is a descendant of the Actor", 0)
        end
        rawget(self, "_binds")[topic] = { handler = handler, parallel = inParallel, owner = owner }
    end
    function methods.BindToMessage(self: any, topic: string, handler: (...any) -> ())
        bindTopic(self, topic, handler, false)
    end
    function methods.BindToMessageParallel(self: any, topic: string, handler: (...any) -> ())
        bindTopic(self, topic, handler, true)
    end
    function methods.SendMessage(self: any, topic: string, ...: any)
        const args = table.pack(...)
        for i = 1, args.n do
            args[i] = copyArgument(args[i], counter)
        end
        sent += 1
        table.insert(deliveries, { frame = frame + latency, target = self, topic = topic, args = args })
    end

    --- Runs the handler a message is bound to, in its phase and for its script. A topic nothing is bound to drops it.
    const function deliver(d: { frame: number, target: any, topic: string, args: { any } })
        const bound = (rawget :: any)(d.target, "_binds")[d.topic]
        if bound == nil then
            dropped += 1
            return
        end
        const thread = coroutine.create(bound.handler)
        owners[thread] = bound.owner
        parallel[thread] = bound.parallel
        const ok, err = coroutine.resume(thread, table.unpack(d.args, 1, (d.args :: any).n))
        if not ok then
            error(`error in a message handler: {tostring(err)}`, 0)
        end
    end

    --- The scheduler.
    task_spawn = function(target: any, ...: any): thread
        const caller = coroutine.running()
        const thread = if type(target) == "thread" then target else coroutine.create(target)
        -- A thread runs in the phase and for the script of the one that starts or wakes it.
        parallel[thread] = parallel[caller]
        if owners[thread] == nil then
            owners[thread] = owners[caller]
        end
        const ok, err = coroutine.resume(thread, ...)
        if not ok then
            error(`error in a spawned thread: {tostring(err)}`, 0)
        end
        return thread
    end
    const taskLib: { [string]: any } = {
        spawn = task_spawn,
        wait = function(_seconds: number?): number
            const thread = coroutine.running()
            table.insert(sleeping, { thread = thread, frame = frame + 1 })
            coroutine.yield()
            return 1 / 60
        end,
        defer = task_spawn,
        desynchronize = function()
            parallel[coroutine.running()] = true
        end,
        synchronize = function()
            parallel[coroutine.running()] = nil
        end,
    }
    mock.task = taskLib

    --- A VM with its own module cache and globals.
    local function makeVm(): Vm
        const vm: Vm = { cache = {}, env = {} }
        const env = vm.env
        for name, value in
            {
                math = math,
                string = string,
                table = table,
                buffer = buffer,
                bit32 = bit32,
                vector = vector,
                coroutine = coroutine,
                utf8 = utf8,
                pcall = pcall,
                xpcall = xpcall,
                error = error,
                assert = assert,
                type = type,
                typeof = typeof,
                tostring = tostring,
                tonumber = tonumber,
                select = select,
                next = next,
                pairs = pairs,
                ipairs = ipairs,
                rawget = rawget,
                rawset = rawset,
                rawequal = rawequal,
                rawlen = rawlen,
                setmetatable = setmetatable,
                getmetatable = getmetatable,
                unpack = table.unpack,
            } :: { [string]: any }
        do
            env[name] = value
        end
        env.os = { clock = os.clock, time = os.time }
        env.print = function(...: any)
            table.insert(printed, table.concat(table.pack(...) :: any, " "))
        end
        env.warn = env.print
        env.task = taskLib
        env.game = mock.game
        env.Instance = Instance
        env.require = function(module: any): any
            return mock.require(module)
        end
        return vm
    end

    mock.game = create("DataModel")
    rawset(mock.game, "Name", "game")
    mainVm = makeVm()
    vms[mock.game] = mainVm

    function vmOf(inst: any): Vm
        const actor = actorOf(inst)
        if actor == nil then
            return mainVm
        end
        local vm = vms[actor]
        if vm == nil then
            vm = makeVm()
            vms[actor] = vm
        end
        return vm :: Vm
    end

    --- Runs the source of a script in `vm` with its own `script` and a `require` that uses the VM's cache.
    function runScript(inst: any, vm: Vm): any
        const source = rawget(inst, "Source")
        if type(source) ~= "string" then
            error(`{rawget(inst, "Name")} has no source`, 0)
        end
        const env = table.clone(vm.env)
        env.script = inst
        env.require = function(module: any): any
            serialOnly("require")
            const cached = vm.cache[module]
            if cached ~= nil then
                return cached.value
            end
            const result = runScript(module, vm)
            vm.cache[module] = { value = result }
            return result
        end
        const bytecode = luau.compile(source)
        const chunk = luau.load(bytecode, `={rawget(inst, "Name")}`, env)
        const results = table.pack(chunk())
        if rawget(inst, "ClassName") == "ModuleScript" and results.n ~= 1 then
            error(`Module code of {rawget(inst, "Name")} did not return exactly one value`, 0)
        end
        return results[1]
    end

    function mock.require(module: any): any
        serialOnly("require")
        const vm = vmOf(module)
        const cached = vm.cache[module]
        if cached ~= nil then
            return cached.value
        end
        const result = runScript(module, vm)
        vm.cache[module] = { value = result }
        return result
    end

    function mock.run(main: (...any) -> ...any, owner: any?)
        local done = false
        const mainThread = coroutine.create(function()
            main()
            done = true
        end)
        owners[mainThread] = owner
        local ok, err = coroutine.resume(mainThread)
        if not ok then
            error(`error in the main script: {tostring(err)}`, 0)
        end
        while not done do
            -- Deliver what is due, in order, then wake what waits for this frame.
            local progressed = false
            local i = 1
            while i <= #deliveries do
                const d = deliveries[i]
                if d.frame <= frame then
                    table.remove(deliveries, i)
                    deliver(d)
                    progressed = true
                    i = 1
                else
                    i += 1
                end
            end
            if done then
                break
            end
            const due: { thread } = {}
            local k = 1
            while k <= #sleeping do
                if sleeping[k].frame <= frame then
                    table.insert(due, sleeping[k].thread)
                    table.remove(sleeping, k)
                else
                    k += 1
                end
            end
            for _, thread in due do
                progressed = true
                task_spawn(thread)
            end
            if not progressed then
                if #deliveries == 0 and #sleeping == 0 then
                    error("the main script waits and nothing can wake it", 0)
                end
                frame += 1
            end
        end
    end

    function mock.phase(): string
        return if parallel[coroutine.running()] then "parallel" else "serial"
    end
    function mock.dropped(): number
        return dropped
    end
    function mock.frame(): number
        return frame
    end
    function mock.setLatency(frames: number)
        latency = frames
    end
    function mock.setHttp(handler: (request: { [string]: any }) -> any)
        httpHandler = handler
    end
    function mock.messages(): (number, number)
        return sent, counter.bytes
    end
    return mock
end

return {
    new = new,
}
