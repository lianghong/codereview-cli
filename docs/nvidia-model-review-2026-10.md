# NVIDIA hosted model review — 2026-10-19

## Scope and evidence

This review checks the four NVIDIA models configured at the start of the
cleanup. It combines first-party NVIDIA catalog pages with authenticated
`GET /v1/models` and small `POST /v1/chat/completions` probes against
[`https://integrate.api.nvidia.com/v1`][api].

Remove Kimi K3 and the scheduled-for-retirement GLM-5.3 endpoint; retain
DeepSeek-V4.1-Flash and GLM-5.3-Flash. This reduces NVIDIA's entries from four
to two and the total registry from 24 to 22.

The catalog returned HTTP 200 with 81 model identifiers. DeepSeek-V4.1-Flash
and both GLM-5.3 variants were listed; Kimi K3 was absent. Invocation and
response headers determine whether a listed model is usable or approaching
retirement. Catalog visibility alone is insufficient.

The page checks included the full NVIDIA page HTML because the main-content
extractor omitted deprecation banners. Search snippets were discovery aids,
not the sole evidence for a retirement decision.

## Kimi K3: remove the retired hosted endpoint

The authenticated completion request for `moonshotai/kimi-k3` returned
HTTP 410. Its response identified the end-of-life time as
`2026-10-15T08:00:00Z` and said the model was no longer available.

The [official NVIDIA page][kimi] independently marks the endpoint deprecated.
Its embedded notice says: "This API will be deprecated on 10/14/2026. It
will no longer be supported after 10/15/2026." The page's `isDeprecated`
value is `true`, and the hosted Prototype example is no longer displayed.

Remove the NVIDIA entry and its provider-specific names. Existing Moonshot
and Bedrock entries offer explicit alternatives in this repository; silently
redirecting an NVIDIA name to either would cross a provider and billing
boundary.

## GLM-5.3: remove before its announced retirement

The authenticated request for `z-ai/glm-5.3` returned HTTP 200 after
75.1 seconds, with this response header:

```text
Deprecation: 2026-10-28T08:00:00Z
```

The [official NVIDIA page][glm] says: "This API will be deprecated on
10/27/2026. It will no longer be supported after 10/28/2026." It currently
has `isDeprecated=false` and still publishes a hosted API example, consistent
with the successful completion.

This endpoint is still live on the review date. Removing it is a proactive
response to NVIDIA's announced sunset, following the repository's existing
[scheduled-retirement policy][registry], rather than a claim that it already
returns HTTP 410. Record that distinction in the removal note and regression
guard. Delete its NVIDIA-specific names instead of redirecting them to Flash,
which is a different tier, or to a billed provider.

## GLM-5.3-Flash: retain

The authenticated request for `z-ai/glm-5.3-flash` returned HTTP 200 after
1.2 seconds with the expected response model. No `Deprecation`, `Sunset`,
or `Link` header was present.

The [official NVIDIA page][glm-flash] still publishes the hosted API example
with the exact configured model identifier. Its `isDeprecated` value is
`false`; no deprecation notice was found in the page HTML.

Retain this entry and the existing NVIDIA default. The parent GLM-5.3
endpoint's retirement does not establish that Flash is retiring too.

## DeepSeek-V4.1-Flash: retain

The authenticated request for `deepseek-ai/deepseek-v4.1-flash` returned
HTTP 200 after 122.6 seconds with the expected response model. No
`Deprecation`, `Sunset`, or `Link` header was present.

The [official NVIDIA page][deepseek] publishes a hosted example using
`deepseek-ai/deepseek-v4.1-flash`. Its `isDeprecated` value is `false`; no
deprecation notice was found in the page HTML. The authenticated catalog
also lists this exact identifier.

Retain this entry. The eight-token completion budget was consumed entirely
by reasoning, ending with `finish_reason=length` (34 prompt tokens and eight
completion tokens). This establishes invocation availability and header
status, not a completed code review or structured-output correctness.

## Limits of these findings

These findings concern NVIDIA's shared hosted API, not the availability of
model weights, GPU runtime support, another provider's deployment, or the
vendor's direct API. NVIDIA's [deployment guide][deployment] distinguishes
hosted preview endpoints from downloadable and self-hosted NIM deployments.
An available model card or deployment page therefore cannot establish that
the shared hosted endpoint remains callable.

No observed sunset on a retained endpoint guarantees future availability.
Repeat catalog, completion, and response-header checks when changing the
NVIDIA registry. A timeout or an HTTP 401/403 response alone does not prove
retirement. This cleanup does not validate other providers.

[api]: https://integrate.api.nvidia.com/v1
[kimi]: https://build.nvidia.com/moonshotai/kimi-k3
[glm]: https://build.nvidia.com/z-ai/glm-5-3
[glm-flash]: https://build.nvidia.com/z-ai/glm-5-3-flash
[deepseek]: https://build.nvidia.com/deepseek-ai/deepseek-v4.1-flash
[deployment]: https://docs.api.nvidia.com/nim/docs/run-anywhere
[registry]: model-registry.md
