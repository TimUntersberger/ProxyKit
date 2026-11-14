# Removing printed text from card images: research plan

## Scope

Target the text printed into a supplied card image: effect text, name, numbers, labels, artist/copyright lines, and vertical text. Keep the translucent SAMPLE watermark as a separately selectable layer in the UI. Work from the supplied pixels only. The source image must remain untouched; export a new result.

The Vivi card is a useful stress test because its text sits on several different surfaces: a partly transparent effect panel, a red nameplate, solid-color symbols, elaborate borders, and artwork. A single fill method is unlikely to preserve all of them equally well.

## Findings

1. **Detection and removal are different problems.** OCR can locate text, but its word boxes cover surrounding art. A removal mask needs to follow glyphs, their outlines, shadows, and compression halos. `ocrs` is a Rust OCR engine with a per-pixel text detector and ONNX-trained models; its authors call it an early preview and currently limit recognition to Latin text. PaddleOCR's current PP-OCRv6 pipeline is a stronger detection candidate and offers mobile detector models and orientation components, though integration into a Rust desktop app needs a separate inference path. Sources: [ocrs](https://github.com/robertknight/ocrs), [ocrs detector model](https://github.com/robertknight/ocrs-models), [PaddleOCR pipeline](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/OCR.html).
2. **Specialized scene-text erasers exist.** ViTEraser publishes pretrained Tiny, Small, and Base weights and inference code. It is worth benchmarking on the card, especially where text crosses artwork. Its published implementation is PyTorch research code; suitability for a cross-platform Rust release and model redistribution need separate checks. Source: [ViTEraser project](https://github.com/shannanyinxiang/ViTEraser).
3. **We already have a general inpainting baseline.** The project contains `research/models/inpainting_lama_2025jan.onnx` (about 93 MB). LaMa was designed for image completion, including repeating structures. It is a plausible local baseline for letter-sized gaps, but we have not tested it for removing all printed card text. Sources: [LaMa paper](https://openaccess.thecvf.com/content/WACV2022/papers/Suvorov_Resolution-Robust_Large_Mask_Inpainting_With_Fourier_Convolutions_WACV_2022_paper.pdf), [official code](https://github.com/advimman/lama).

## Recommended experiment

1. On the original 1x Vivi PNG and the supplied 2x version, run text detection at multiple scales plus 90-degree rotations. Save detection overlays and allow human correction. Check each category separately: effect text, name, stat numerals, tiny edge text, and the SAMPLE word. Do not silently erase decorative symbols or borders.
2. Build a tight stroke mask inside each accepted detection. Include colored outlines and shadows, then expand just enough to cover antialiasing and halos. Display the mask at full resolution. Keep an editable brush for missed or falsely selected pixels.
3. Compare three fills on identical masks: (a) local color/gradient or line continuation for simple card graphics, (b) the existing LaMa ONNX model for textured regions, and (c) ViTEraser as a research benchmark. Composite each candidate only inside its approved repair mask. Try the current FLUX workflow only for regions where these methods visibly fail; large generative fills risk changing art and card geometry.
4. Review crops at 100% and card size. Record text remnants, border continuity, color seams, changed artwork, and exact pixels changed outside the mask. For quantitative testing, put synthetic text over originally text-free patches from unrelated supplied images and compare against the known clean patches. Real printed text has no known ground truth.

## Decision for the MVP

Build **detect → inspect/edit mask → choose fill per region → composite → compare**. Start with an automatic suggestion and a manual correction step. Keep recognition optional: removal needs text location more than the decoded words. Do not bake Vivi coordinates or card templates into the detector. Once a small set of unrelated cards has been evaluated, choose the detector and fill engine for the Rust/Tauri implementation.

The missing image beneath opaque print cannot be recovered exactly from one raster card. Flat backgrounds and repeating patterns can often be reconstructed convincingly; text over intricate artwork remains an estimate. An upscaled input may make masks and visible seams easier to handle, but does not restore hidden source detail.

## Automatic detection preview

`Highlight-Card-Text.ps1` runs `highlight_card_text.py` with the isolated `text_detect_venv` research environment. It uses RapidOCR's local PP-OCRv6 detector and recognition models, with upright and sideways passes. It saves a highlighted PNG and a JSON file of detected polygons beside the input by default. Red indicates OCR-recognized text; orange indicates lower-confidence detector candidates. This is a detection preview, not a removal mask: it can miss stylized letters and flag decorative shapes.

```powershell
& .\research\Highlight-Card-Text.ps1 'C:\path\to\card.png'
```

On a new machine, create `research/text_detect_venv` with Python 3.11 and install `rapidocr onnxruntime` into it. The Python script can also be run directly with `--output` and `--threshold`.

### First inpainting baseline

`probe_text_removal.py` filled the 15 red polygons from the 2x Vivi upright-only detection with the local LaMa ONNX model, using 13 overlapping 512px patches. It changed zero pixels outside the polygon mask, but the result is poor: the wide rules-text boxes erase clothing, flowers, and card decoration, and the false-positive diamond is damaged. Output: `archive/root-images/vivi_text_box_lama_probe.png`; mask: `archive/root-images/vivi_text_box_lama_probe_mask.png`. This confirms that detection polygons must first be converted to tighter glyph/outline masks. OCR recognition provides per-character boxes that can bound an automatic foreground segmentation step; the resulting mask needs visual evaluation before another inpainting comparison.

`-HorizontalOnly -RecognizedOnly -TightBoxes` now draws OCR character boxes and narrows the vertical band of long text lines using an image-edge projection. On the 2x Vivi preview it retained 15 recognized regions and 222 character boxes. Their combined area is 241,497 pixels versus 359,978 pixels for the original line polygons, a 32.9% reduction. Output: `archive/root-images/vivi_text_character_boxes_trimmed.png` and matching JSON. These are still rectangles around letters, not the ink pixels themselves; they should not be used directly as a final inpainting mask.

### Removal probes on the 2x Vivi image

- LaMa with character rectangles (`archive/root-images/vivi_text_character_lama_probe.png`) left many letter fragments and distorted details under the rules text. A color-enrichment mask (`archive/root-images/vivi_text_color_mask.png`) and OpenCV Telea/LaMa fills also left fragments or strong smears. These are unsuccessful outputs.
- A single-image FLUX.2 Klein crop, prompted to remove the printed rules text, produced a clean center crop without lettering (`archive/root-images/vivi_text_flux2_single_probe_raw_crop.png`). The image-plus-mask-reference prompt kept the lettering, so the single-image instruction worked better in this probe.
- `probe_text_flux2_tiled.py` grouped the 15 red OCR regions into four horizontal bands and ran seven 640x512 local model crops. It composited within dilated detection polygons and changed zero pixels outside that mask. `archive/root-images/vivi_text_flux2_tiled_probe.png` removes the detected rules and name text much more completely than LaMa. It reconstructs dress folds and card motifs imperfectly, creates some seams, and leaves all text that OCR did not mark red (including SAMPLE, vertical print, and some labels). This is a feasibility result, not a reliable general-purpose text remover.

### UI follow-up on Saint Gunko

The first user result used a `*_text_highlighted.png` detection preview as input, so red boxes were painted into the source. The UI now resolves a selected detection preview to the unmarked source recorded in its JSON. The first raw-source repaint replaced the white `5000` area with blue and damaged the adjacent SPECIAL emblem. The revised pipeline detects nearly uniform white surroundings and fills those selected text regions directly; it preserves compact nonuniform top emblems. It also adds vertical crop coverage for tall text groups. `archive/root-images/gunk_text_removed_fixed3.png` is the horizontal-only result. `archive/root-images/gunk_text_removed_vertical.png` includes a filtered vertical OCR pass and removes the right-edge print as well. Both runs changed zero pixels outside their repair masks. The numeral `4`, the icon `5`, and the SPECIAL emblem remain because they were undetected or deliberately preserved. Some reconstructed art within the rules-text area is estimated and can differ from the source.

### Japanese Edward Newgate card

On `HOuCaWdbAAAbOhf.jpg`, OCR located Japanese rules and title text, but the original workflow sent the white `8000` header and blue nameplate to FLUX. It repainted the header with unrelated colors and introduced decorative marks in the blue bar. The revised `probe_text_flux2_tiled.py` estimates a smooth color plane from clean pixels around low-variation text regions and composites only inside their masks. It preserves compact emblems and small detailed symbols. `archive/root-images/japanese_text_smooth_probe_safe.png` is the resulting preview: the header and nameplate are much cleaner, while the rules-text area still contains model-estimated anatomy and visible band seams. The source artwork under opaque Japanese rules text cannot be recovered exactly from this raster card.
