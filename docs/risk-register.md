# Risk Register — AOI Defect Detection MVP

Mirror of GitHub Project [risks view](https://github.com/users/hsuani/projects/1/views/4).

| ID | Risk | Likelihood | Impact | Mitigation | Status |
|---|---|---|---|---|---|
| R-1 | LaunchPad lab approval > 3 days | High | Med | Brev signup credit fallback | Open |
| R-2 | DeepStream 7.x API breaking changes | Med | High | Pin NGC 7.0 container | Open |
| R-3 | INT8 mAP degrades > 10% on real defects | Med | High | FP16 fallback as primary | Mitigated |
| R-4 | TRT engine non-portable across GPU arch | Cert. | Low | Per-target rebuild | Accepted |

Each entry tracked as GitHub issue with `type:risk` label.
