# Blue Character card layout reference

Measured from the user-supplied `017-049.png` (1000 × 1398 pixels). Coordinates
are pixel positions in that source image, with origin at its upper-left corner.
The generator uses the same output dimensions.

| Element | Reference bounds or anchor |
| --- | --- |
| Rounded white outer card | `(0, 0)` to `(999, 1397)`; corner radius about 43 px |
| Blue artwork rail | left x≈35–51; right x≈949–960; top y≈43–55 on left, y≈112–122 on right |
| Cost disc | center `(89, 91)`, radius ≈67 px |
| Power header | white angled opening from x≈594, y≈43 to x≈645, y≈113; power centered x≈756, baseline y≈96 |
| Special badge | center around `(921, 57)`, about 86 × 94 px including its lower point |
| Rules panel | x≈63–937, y≈850–1150, corner radius ≈7 px |
| First effect line | badge top y≈866, text baseline y≈894 |
| Blue class tab | x≈325–674, top y≈1167; class baseline y≈1198 |
| Name | centered x=500, baseline y≈1275 |
| Type capsule | x≈305–695, y≈1294–1339 |
| Footer | x≈43–957, bottom y≈1358 |
| Lower-left color icon | polygon spanning x≈24–124, y≈1269–1381 |
| Card number and rarity | right aligned near x≈874, baseline y≈1342 |

Measured visible color samples from the source (RGB at the named coordinate):

| Point | RGB | Hex | Use |
| --- | --- | --- | --- |
| `(40, 700)` | `(8, 142, 198)` | `#088ec6` | Bright blue rail |
| `(45, 1170)` | `(0, 137, 198)` | `#0089c6` | Blue near footer shoulder |
| `(200, 1220)` | `(1, 134, 191)` | `#0186bf` | Upper-left footer |
| `(800, 1220)` | `(3, 121, 185)` | `#0379b9` | Upper-right footer |
| `(200, 1340)` | `(0, 129, 190)` | `#0081be` | Lower-left footer |
| `(800, 1340)` | `(2, 113, 185)` | `#0271b9` | Lower-right footer |

These are samples of a rendered, antialiased raster card, not a complete source
template. The exact typefaces, logo artwork, gradients, and covered pixels cannot
be recovered unambiguously from this one image. The current canvas implementation
matches measured positions but recreates those details with browser fonts and
vector drawing. A pixel-identical result requires original frame and font assets.
