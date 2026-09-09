# DeepSeek and Z.AI upstream model review — 2026-09-09

## Scope

This review checks the direct DeepSeek and Z.AI APIs against first-party model,
pricing, migration, and release documentation. It records the evidence used to
update `codereview/config/models.yaml`; it does not treat re-hosted NVIDIA or
Bedrock model catalogs as substitutes for the vendors' own APIs.

## DeepSeek

### Current upstream models

DeepSeek's official pricing table lists three current API identifiers:

- `deepseek-v4-pro` — model version DeepSeek-V4-Pro-0813.
- `deepseek-v4-flash` — model version DeepSeek-V4-Flash-0731.
- `deepseek-v4-flash-vision-exp` — an experimental multimodal variant.

All three advertise a 1M-token context, a 384K maximum output, JSON output,
tool calls, the Responses API, and both thinking and non-thinking modes
([DeepSeek models and pricing](https://api-docs.deepseek.com/quick_start/pricing/)).
DeepSeek's changelog confirms that the stable Pro and Flash API names float to
the current GA versions without changing the caller's model string
([DeepSeek changelog](https://api-docs.deepseek.com/updates/)).

### Pricing correction

The previous registry rates predated DeepSeek's August 2026 peak/off-peak
schedule. The official table now charges:

| Model | Peak input | Peak output | Off-peak input | Off-peak output |
| --- | ---: | ---: | ---: | ---: |
| V4-Pro | $1.32 | $3.96 | $0.66 | $1.98 |
| V4-Flash | $0.44 | $1.32 | $0.22 | $0.66 |

Rates are per million tokens. Peak hours are 01:00–04:00 and 06:00–10:00 UTC,
Monday through Friday; all other times are half-price
([DeepSeek models and pricing](https://api-docs.deepseek.com/quick_start/pricing/)).
The CLI stores peak rates because its pricing schema has no time-of-day tier and
under-estimating a peak run is worse than over-estimating an off-peak run.

### Registry decision

Keep `deepseek-v4-pro` and `deepseek-v4-flash`: they remain the latest stable
text models and are appropriate for source review. Update their rates to peak
pricing. Keep the conservative 16K generation cap; the published 384K is an API
ceiling, not a requirement to budget that much output for one review.

Do not add `deepseek-v4-flash-vision-exp`. DeepSeek calls it experimental and
states that its pure-text capabilities are on par with V4-Flash
([DeepSeek changelog](https://api-docs.deepseek.com/updates/)). This CLI sends
source text, not screenshots or rendered interfaces, so the experimental entry
would duplicate the stable Flash model without using its distinguishing
capability.

## Z.AI

### Current upstream models

Z.AI identifies GLM-5.3 as its latest flagship. It is text-only, has a 1M-token
context and 128K maximum output, and always runs reasoning with `low`, `high`, or
`max` effort (`max` by default). Function calling and structured output are
advertised capabilities
([GLM-5.3 model page](https://docs.z.ai/guides/llm/glm-5.3)). Z.AI reports that
post-training delivers a 50% improvement over GLM-5.2 on Z.ai Code Bench while
using the same base model.

GLM-5.3-Flash is the current low-cost sibling and the first native multimodal
model in the GLM-5 family. Its model code is `glm-5.3-flash`; it has 320B total
and 18B active parameters, a 1M context, always-on reasoning, function calling,
structured output, and text/image/video/file input
([GLM-5.3-Flash model page](https://docs.z.ai/guides/vlm/glm-5.3-flash)).

### Pricing

Z.AI's official pricing table lists:

| Model | Input | Cached input | Output |
| --- | ---: | ---: | ---: |
| GLM-5.3 | $1.40 | $0.26 | $4.40 |
| GLM-5.3-Flash list price | $0.15 | $0.03 | $0.50 |
| GLM-5.3-Flash temporary price | $0.075 | $0.015 | $0.25 |

Rates are per million tokens
([Z.AI pricing](https://docs.z.ai/guides/overview/pricing)). The registry stores
Flash's list rate because the discounted price is explicitly temporary.

### API and parameter behavior

Z.AI's migration guide says GLM-5.3 defaults to `temperature=1.0`,
`top_p=0.95`, and always-on thinking; disabling thinking returns an error. It
also confirms the 1M context and 128K output ceiling
([migration guide](https://docs.z.ai/guides/overview/migrate-to-glm-new)). The
provider retains the pay-as-you-go OpenAI-compatible base URL. Both the existing
`/api/paas/v4/models` route and the Coding Plan's `/api/coding/paas/v4/models`
route were probed without credentials on 2026-09-09 and returned the expected
HTTP 401 authentication response rather than 404.

### Registry decision

- Replace GLM-5.2 with GLM-5.3. They have identical published pricing and
  limits, while 5.3 is the stated successor with stronger post-training.
- Add GLM-5.3-Flash as the budget model. Its multimodal support is recorded,
  but the current review pipeline still sends text only.
- Move generation-neutral aliases (`glm`, `zai-glm`, `glm-zai`) to GLM-5.3.
  Delete version-explicit 5.2 names instead of silently redirecting them.
- Use conservative 32K output caps for both entries.
- Set `supports_tool_use: false` for both. They are always-thinking models, and
  the repository requires a live review proving LangChain's forced tool choice
  works while thinking before enabling tool-based structured output. Vendor
  capability claims alone do not meet that empirical bar.
