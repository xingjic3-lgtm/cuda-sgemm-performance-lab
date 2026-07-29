# Phase 5 Nsight Compute evidence

The repository keeps the extracted evidence in [`metrics.csv`](metrics.csv). Raw
`.ncu-rep` files are intentionally excluded from Git because they are large,
tool-version-specific binary artifacts.

## Raw report manifest

| Local file | Bytes | SHA-256 |
|---|---:|---|
| `phase5_k10_config21_4096_before.ncu-rep` | 34,631,012 | `5B35DD21509219F72748781D9B517306F97BD87C72948EC96994046D5AF7DAE2` |
| `phase5_k11_config21_4096_before.ncu-rep` | 43,118,811 | `DCDB33F92C7AE033518326F828342317CB12E1CEE236E7A7ACE6AD268B680C62` |
| `phase5_k11_config21_split31_4096.ncu-rep` | 524,471 | `E69097884EF43E6B65FAFF233C62446DFFD6115F82436E6B6691EFD781589EB1` |
| `phase5_k11_config38_4096_after.ncu-rep` | 511,549 | `133F24A5B4110348B9FBE6D64BFA460FDCFAF9A387D878504556DE40E0CED678` |

These files were collected one directory above the repository. Before
publishing a release, they may be attached as optional release assets; copying
them into normal Git history is not recommended.

## Reproduction contract

- Ordinary execution decides whether a change is faster.
- Nsight Compute explains where issue opportunities were lost.
- A modified kernel is accepted only after an ordinary rerun and a comparison
  of the same diagnostic metrics.
- Nsight replay duration is never compared with ordinary elapsed time.

Representative collection:

```powershell
ncu --kernel-name regex:sgemmDoubleBuffering `
  --launch-skip 1 --launch-count 1 --set full `
  -f -o ..\phase5_k11_config21_4096_before `
  .\build\sgemm.exe 11 4096 4096 4096
```

`--launch-skip 1` skips the first matching kernel launch, and
`--launch-count 1` profiles the next matching launch once. A full report may
replay that launch many times to gather incompatible counter sets.

