# Standard MAT Data Format

Required variables:

| Variable | Shape | Meaning |
|---|---:|---|
| `anchors` | `A × 2` | Anchor coordinates in metres |
| `link_pairs` | `L × 2` | One-based or zero-based anchor indices |
| `time_s` | `T` | Common time axis |
| `delay_grid_ns` | `B` | Excess-delay grid |
| `trajectory_xy` | `T × 2` | Ground-truth location |
| `tof_total_ns` | `T × L` | Ground-truth bistatic ToF |
| `tof_los_ns` | `L` | Direct anchor-to-anchor ToF |
| `cir_background` | `L × B` | Background CIR |
| `var_background` | `L × B` | Background variance |
| `cir_dynamic` | `T × L × B` | Dynamic CIR |
| `var_dynamic` | `T × L × B` | Dynamic variance |

The loader verifies that the ground-truth ToF agrees with anchor geometry.
