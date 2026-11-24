--!native
-- The entry point for callers: opens a container, loads the model for this host and starts conversations.

const budget = require("@src/runtime/budget")
const chat = require("./chat")
const container = require("@src/container")
const coordinator = require("@src/parallel/coordinator")
const host = require("@src/host")
const model = require("@src/model")
const plan = require("@src/model/plan")
const probe = require("@src/kernels/probe")
const runnerApi = require("@src/model/runner")
const prefix = require("./prefix")
const sampler = require("@src/sampling/sampler")
const session = require("@src/model/session")
const tokenizer = require("@src/tokenizer")
const modelTypes = require("@src/model/types")
const samplingTypes = require("@src/sampling/types")
const types = require("./types")

type Engine = types.Engine
type Options = types.Options

const DEFAULT_CONTEXT = 2048
const DEFAULT_CHUNK = 32

--- Opens the model and loads its weights.
const function new(options: Options): Engine
    const c = container.open(options.source)
    const context = options.context or DEFAULT_CONTEXT
    const chunk = options.chunk or DEFAULT_CHUNK
    const kv = options.kv or "f32"
    if kv ~= "f32" and kv ~= "q8" then
        error(`the KV cache format must be "f32" or "q8", not {kv}`, 0)
    end
    const caps = options.caps or { vectorRead = probe.run(options.host.clock).vectorRead }
    const layout = options.kvLayout or model.kvLayoutFor(c.header.config.headDim, caps, kv)
    const estimate = plan.estimate(c, context, chunk, kv, layout)
    const transports = options.transports
    const memoryLimit = options.memoryLimit
    if transports ~= nil and c.header.config.layerTypes ~= nil then
        error("a model with short convolution layers cannot be divided among workers", 0)
    end
    if memoryLimit ~= nil and transports == nil then
        plan.check(estimate, memoryLimit)
    end
    const limit = budget.new(options.host, options.slice or 0)
    local m: modelTypes.Model? = nil
    local runner: modelTypes.Runner
    if transports ~= nil then
        runner = coordinator.new(c, transports, { context = context, chunk = chunk, budget = limit })
    else
        const loaded = model.load(c, caps, context, nil, kv, options.kvLayout)
        m = loaded
        runner = runnerApi.single(loaded, chunk, limit)
    end
    const shared: types.Core = {
        host = options.host,
        config = c.header.config,
        model = m,
        runner = runner,
        tokenizer = tokenizer.new(c),
        budget = limit,
        chunk = chunk,
        seed = options.seed or 1,
        prefixes = nil,
    }
    const prefixes = options.prefixes
    const capacity = if prefixes ~= nil then prefixes else 8
    if capacity > 0 then
        shared.prefixes = prefix.new(runner, capacity)
    end
    const engine: Engine = {
        core = shared,
        caps = caps,
        kv = kv,
        context = context,
        chat = function(_self: Engine, chatOptions: types.ChatOptions?): types.Chat
            return chat.new(shared, chatOptions)
        end,
        tokenize = function(_self: Engine, text: string): { number }
            return tokenizer.encode(shared.tokenizer, text, nil, true)
        end,
        detokenize = function(_self: Engine, ids: { number }): string
            return tokenizer.decode(shared.tokenizer, ids)
        end,
        generate = function(
            _self: Engine,
            ids: { number },
            maxTokens: number,
            sampling: samplingTypes.Options?
        ): { number }
            const sess = runner:open()
            const smp = sampler.new(shared.seed)
            const history: { number } = table.clone(ids)
            local logits: buffer? = nil
            local at = 1
            while at <= #ids do
                const count = math.min(chunk, #ids - at + 1)
                logits = runner:forward(sess, ids, at, count, at + count > #ids, nil)
                at += count
            end
            const out: { number } = {}
            while #out < maxTokens and sess.length < sess.context do
                const last = logits
                assert(last ~= nil, "the prompt produced logits")
                const id = sampler.sample(smp, last, shared.config.vocab, history, sampling or {})
                table.insert(out, id)
                table.insert(history, id)
                logits = runner:forward(sess, { id }, 1, 1, true, nil)
            end
            runner:close(sess)
            return out
        end,
        session = function(_self: Engine, tokens: number?): modelTypes.Session
            const loaded = m
            if loaded == nil then
                error("the layers are held by workers, there is no session in this VM", 0)
            end
            return session.new(loaded, tokens or chunk)
        end,
        close = function(_self: Engine)
            runner:shutdown()
        end,
        memory = function(_self: Engine): modelTypes.Estimate
            return estimate
        end,
    }
    return engine
end

return {
    new = new,
}
