# docs.glacis.io — labs + brand alignment (2026-09-27)

## IA choice

**Keep** the origin/main activation ladder (`/start/`, `/connect/`, `/verify/`,
`/reference/`) and **add** company-wide sections required by labs + brand:

| Section | Role |
| --- | --- |
| Start / Connect / Verify / Reference | Honest SDK + portal onboarding (preserved from main) |
| Runtime | Customer-facing product claim boundary (`CLAIM.md` paraphrase) |
| OVERT | Orientation only; normative text stays at overt.is |
| OVERT-as-Code | Preview — not GA |
| Concepts | Thought-leadership hubs |

**Rejected:** reverting to SDK-only docs; replacing Start/Connect with the June
scaffold’s `/sdk/python/*` tree as primary (PyPI links still redirect into
Connect). Live Cloudflare tree had drifted toward the June company-wide scaffold
while `origin/main` had the more careful Start/Connect honesty — this branch
merges both.

## Vocabulary

- Human-facing: **record** (matches glacis-web-prod).
- Spec / wire / SDK types: **receipt** where that is the actual name.
- Homepage notes the dual vocabulary explicitly.

## Claim flags (do not exceed)

- No EU AI Act / ISO / SOC “done.”
- Default customer-hosted bundle: OVERT Level 1 Core / **AAL-3** ceiling.
- Not AAL-4 / not “evidence-grade.”
- Mediated scope: records attest governed paths only.
- Offline SDK `oatt_…` ≠ conforming OVERT document.
- OVERT-as-Code + OSCAL export: **Preview**.
- Online/witnessed attestation: not GA.
- Utah / CHAI: stewardship/news only; no “partnership,” no certification claims.
- No secrets from labs runbooks.

## Follow-ups (not in this PR)

1. Full Start/Connect prose pass for remaining “receipt” in human sentences on deep pages.
2. Port main’s `verify/what-a-check-proves` examples to say “record” in gloss copy while keeping field tables accurate.
3. Confirm hosted-Notary PrivateLink topology vs co-resident AAL-3 wording with Joe before changing `CLAIM.md` paraphrase.
4. Decouple docs from `glacis-python` repo (still deferred).
5. Coordinate merge with open PR #11 (`fix/docs-sitemap-xml`) — `_redirects` sitemap rule is identical on purpose.
6. SDK README still says “audit logs” / online-soon — separate package docs PR.

## Preview

```bash
cd docs
npm install
npm run build
npm run preview   # or: npx astro preview
```

Do **not** production-redeploy until Joe OKs.
