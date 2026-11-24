-- A small test harness with no host imports. A suite is a module returning a function that takes the harness and
-- registers cases. `run.luau` runs every suite and exits non zero when a case fails.

--- What a suite sees.
export type Harness = {
    case: (name: string, body: () -> ()) -> (),
    skip: (name: string, reason: string) -> (),
    equal: (actual: any, expected: any, message: string?) -> (),
    truthy: (value: any, message: string?) -> (),
    near: (actual: number, expected: number, tolerance: number, message: string?) -> (),
    fails: (body: () -> (), message: string?) -> (),
}

--- The outcome of a run.
export type Totals = {
    passed: number,
    failed: number,
    skipped: number,
}

--- A suite registers its cases on the harness.
export type Suite = (h: Harness) -> ()

--- Runs one suite under `name`, printing failures and skips, and adds its counts to `totals`.
const function runSuite(name: string, suite: Suite, totals: Totals)
    local passed, failed, skipped = 0, 0, 0

    const function fail(message: string)
        error(message, 3)
    end

    const h: Harness = {
        case = function(caseName: string, body: () -> ())
            const ok, err = pcall(body)
            if ok then
                passed += 1
            else
                failed += 1
                print(`  FAIL {name}: {caseName}\n    {tostring(err)}`)
            end
        end,
        skip = function(caseName: string, reason: string)
            skipped += 1
            print(`  skip {name}: {caseName} ({reason})`)
        end,
        equal = function(actual: any, expected: any, message: string?)
            if actual ~= expected then
                fail(`{message or "values differ"}: got {tostring(actual)}, expected {tostring(expected)}`)
            end
        end,
        truthy = function(value: any, message: string?)
            if not value then
                fail(message or "expected a truthy value")
            end
        end,
        near = function(actual: number, expected: number, tolerance: number, message: string?)
            if not (math.abs(actual - expected) <= tolerance) then
                fail(`{message or "values differ"}: got {actual}, expected {expected} within {tolerance}`)
            end
        end,
        fails = function(body: () -> (), message: string?)
            if pcall(body) then
                fail(message or "expected an error")
            end
        end,
    }
    suite(h)
    totals.passed += passed
    totals.failed += failed
    totals.skipped += skipped
    print(`{if failed > 0 then "FAIL" else "ok  "} {name}: {passed} passed, {failed} failed, {skipped} skipped`)
end

return {
    runSuite = runSuite,
}
