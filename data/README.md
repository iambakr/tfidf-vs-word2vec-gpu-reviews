# Data Directory

## `sample_reviews.csv`

**⚠️ These 15 reviews are SYNTHETIC.** They were written to illustrate the schema and
the dominant themes found in the real corpus (coil whine, dead-on-arrival units,
refund/RMA friction, DLSS/ray-tracing praise, thermal & acoustic performance), so the
analysis scripts in this repository can be demonstrated end-to-end without redistributing
Amazon content.

The real study corpus — **2,989 U.S. Amazon reviews** of NVIDIA RTX 40 series cards
(ASUS / GIGABYTE / MSI, Oct 2022 – Apr 2025) — is **not redistributed** here out of
respect for Amazon's terms of service. All results reported in
[`PAPER.md`](../PAPER.md) and [`RESULTS.md`](../RESULTS.md) come from the real corpus.

## Schema

| Column | Type | Description |
|---|---|---|
| `review_id` | string | Synthetic identifier (`S001`–`S015`) |
| `brand` | string | Board partner: ASUS / GIGABYTE / MSI |
| `chipset` | string | NVIDIA RTX 40 series chipset |
| `vram_gb` | int | Video memory in GB |
| `star_rating` | int | 1–5 stars |
| `label` | string | `BAD` (1–3 stars) / `GOOD` (4–5 stars) — the classification target |
| `review_text` | string | Review body |

## Real-corpus reference statistics

| Property | Value |
|---|---|
| Final sample | 2,989 reviews |
| Class ratio (GOOD : BAD) | ≈ 4.76 : 1 |
| Training corpus | 63,309 tokens / 6,532 unique terms |
| Brand mix | GIGABYTE 40.1% · ASUS 33.5% · MSI 26.5% |
| Top negative bigram | "coil whine" (170 mentions — #1 overall) |
