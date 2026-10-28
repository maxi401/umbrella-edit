# Fashion archive prototype

This is a local, reproducible baseline for the sample archive in `~/Desktop/fashionArchive`.

## Open the app

```bash
cd "/Users/yanhuayao/Desktop/教职申请资料/fashion_archive_prototype"
python3 app.py
```

Then open `http://127.0.0.1:8765` in a browser. Stop the local server with `Ctrl+C` in the Terminal window.

The app is responsive for laptop and phone-sized screens. By default it binds to this computer only. To test from phones on the same trusted Wi-Fi, start it with `python3 app.py --host 0.0.0.0`; then open `http://<this-Mac’s-local-IP>:8765` in Safari on each phone. Find the Mac’s local IP in **System Settings → Wi-Fi → Details**. The current prototype has no sign-in, so keep LAN testing on a trusted network and stop the server when finished. Devices on another network cannot reach this local address.

## App features

- Browse the imported sample images with category filters, search, and sorting.
- Open on a daily style dashboard with a rule-based archive rotation pick, saved-outfit counts, logged-wear counts, and a weekly favorite once wear logs exist.
- Edit the provisional record type; add brand, price, purchase date/place, ownership status, notes, and tags.
- Mark an archived outfit photo as worn and save a dated wear log.
- Upload a new photo and classify it as OOTD, owned/bought candidate, interest, or inspiration.
- Store editable fields and wear logs in the local SQLite database `fashion_archive.sqlite3`; new uploads are stored in `uploads/`.

The original sample folder is read as supplied. Back up `fashion_archive.sqlite3` and `uploads/` if you want to preserve new app entries.

The daily pick is deliberately a transparent heuristic, not a trained CV recommender. It rotates among titled OOTD posts, preferring fewer logged wears and avoiding looks worn in the previous seven days. Until real OOTD/wear logs are added, it is an archive prompt; it does not know the current weather, occasion, or whether every pictured garment is still owned.

## Run

```bash
python3 analyze_archive.py
```

To use another input/output folder:

```bash
python3 analyze_archive.py --source /path/to/fashionArchive --output /path/to/output
```

## Current pipeline

1. Recursively catalog supported image files and derive a provisional record type from the parent folder.
2. Parse Xiaohongshu export filenames for title, slide number, and author label when the filename follows the observed pattern.
3. Extract file metadata and basic whole-image visual descriptors: average RGB/HSV, luminance entropy, edge energy, and color variation.
4. Write a row-level `image_catalog.csv` and aggregate `summary.json`.

All processing is local. No network calls, model downloads, face recognition, identity inference, or brand/item recognition are performed.

## Current output for the supplied sample

- 113/113 images readable.
- 57 OOTD, 27 inspiration, 26 interest candidates, 3 owned candidates based on folder names.
- 19 exported post titles contain multiple images/slides.

These counts are sample inventory, not confirmed ownership or behavioral statistics. Whole-image color/texture measures are affected by backgrounds, lighting, screenshots, and text overlays. Filename title/slide fields are source hints and should be reviewed.

## Product data model suggested by the sample

- `source_post`: source, title/caption, source date, original post URL if known.
- `source_image`: image path, slide order, image metadata, review status.
- `wardrobe_item`: category, brand, price, purchase date/place, ownership state, item image.
- `outfit`: worn date, context, linked wardrobe items, outfit image.
- `brand_reference`: brand/designer/collection, reference type, notes and visual tags.

Keep post images separate from confirmed wardrobe items. One post can contain multiple images, outfits, or references; an image can suggest a product without proving it was purchased or owned.

## Garment segmentation prototype

The **Visual analysis** view shows a first segmentation pass for tops, dresses, bottoms, bags, hats, and accessories. It uses the pretrained `fashn-ai/fashn-human-parser` model locally; it has not been trained on this personal archive. The broad “accessory” group currently combines belts, scarves, glasses, and jewelry.

To regenerate the results, use a Python environment with PyTorch, Transformers, NumPy, and Pillow installed:

```bash
HF_HUB_OFFLINE=1 /opt/anaconda3/bin/python segment_archive.py
```

The run reads `output/image_catalog.csv` and writes reviewable masks/crops, overlays, `output/segmentation/segmentation_results.json`, and `output/segmentation/segmentations.csv`. Use `--incremental` to keep previous image results and process only new or changed files; model weights must already be cached locally for offline mode.

The app now indexes new files when it starts and queues local segmentation for each photo added through **Add a photo**. New uploads are kept with the current results, then the browser refreshes their cutouts when inference finishes. This uses the same cached pretrained model and may take time on a CPU-only computer. Restart the local app after updating its code for the upload worker to take effect. Batch-imported archive images still need to be added to `output/image_catalog.csv` by running `analyze_archive.py` before `segment_archive.py --incremental`.

For this sample, 113 images produced 305 category detections. Category counts are images with a prediction, not verified item counts: top 95, dress 50, bottom 69, bag 37, hat 17, accessory 37. Treat the masks and confidence values as suggestions. A visual spot-check found a plush toy incorrectly labeled as clothing, so detections need human review, especially on collages, runway scenes, small subjects, and partly obscured items.

The model and images run on this computer; segmentation does not upload the archive. The model card identifies its categories and limitations: [FASHN Human Parser](https://huggingface.co/fashn-ai/fashn-human-parser). Its underlying [NVIDIA SegFormer license](https://github.com/NVlabs/SegFormer/blob/master/LICENSE) is limited to non-commercial research and evaluation. This is suitable for a private prototype evaluation, but the model cannot be carried into a commercial agency product without replacing it with a commercially permitted model or obtaining permission.

## Next analysis steps

1. Let a person confirm, correct, or reject each predicted garment group; save those reviews as labels.
2. Add OCR/caption extraction only as editable suggestions, preserving the original source text and confidence.
3. Once there are enough reviewed masks, fine-tune a model on the real photos and categories; until then, use the pretrained parser as a baseline.
4. Build recommendation scores from explicit wear logs and human-confirmed items first; use visual novelty as one optional signal, not a proxy for taste or suitability.

## Outfit match evidence

The home screen draws top-and-bottom combinations from different OOTD posts, and explains which past outfit has a similar coarse color contrast. For variety it prefers pieces absent from the five most recent draws and avoids the last 50 combinations where alternatives exist; this history is saved in the current browser. This avoids selecting both cutouts from the same photo or post; since individual garments are not linked across photos, it cannot guarantee a pairing has never been worn before. The home screen also selects a saved inspiration photo without a confident garment detection by comparing whole-image average color to the garment colors. This is a palette cue; backgrounds and detection errors can affect it.

Look Ideas compares detected garments in inspiration photos with the same category among confirmed owned pieces and closet candidates, using average garment color. Candidates are labeled until ownership is confirmed. These color comparisons are approximate and do not identify garment shape, brand, or taste. New uploads have whole-image color measured during segmentation, so newly added inspiration photos can participate.

## Hosting for independent phone use

The local app is not yet ready for private online hosting. A hosted version needs authenticated access for the intended users, persistent storage for records and photos, and a hosted segmentation service. Its current Python model runs on this Mac and cannot simply be placed in a Cloudflare Worker. Migrate existing photos only into private storage, and check that the segmentation model's license permits the intended use. Until those pieces are implemented and deployed, the iPhone uses the local app server described above.

## Weather and occasion context

The styling ideas accept a city name and request current conditions plus today's high/low and precipitation chance from Open-Meteo. The city is stored in this browser and the forecast is cached for one hour; the app does not request device geolocation. Each idea gets the same weather cue (for example, possible rain or a cold-day extra layer) and a date/occasion cue. China’s common lunar festivals are detected with the browser's Chinese calendar; several fixed-date international occasions are included. A manual plans selector covers work, dinner/date, celebrations, and travel.

Weather and occasion cues are explanatory context. They do not reorder outfit pairs because this archive has no reliable garment warmth, material, comfort, or formality labels. Open-Meteo's free endpoint is limited to non-commercial use; use a commercial plan if this becomes part of a commercial agency product ([pricing](https://open-meteo.com/en/pricing), [terms](https://open-meteo.com/en/terms)).

### Storage preparation for a future host

The server can use `FASHION_DATA_DIR` for its SQLite database, catalog, upload files, and segmentation output, and `FASHION_SOURCE_DIR` for imported source photos. Without these variables the original local paths are used. On a hosted server these locations must be backed by persistent private storage, and the existing archive must be migrated separately. This path configuration alone does not provide authentication, backups, deployment, model installation, or an archive migration. Do not expose the current server to the internet until those pieces are in place.

### Render deployment preparation (not yet verified or deployed)

`Dockerfile`, `Caddyfile`, and `start-hosted.sh` prepare a single-instance Docker service. Render must terminate HTTPS, mount a persistent disk at `/data`, and provide `FASHION_USERNAME` and a Caddy bcrypt `FASHION_PASSWORD_HASH` as secret environment variables. The Python server listens on loopback behind the password-protected proxy. The service refuses to start without the disk, credentials, and Render hostname. The app enforces the Render hostname as the origin for state-changing requests. Use a long randomly generated password and share it securely; basic authentication does not provide individual accounts or built-in rate limiting. Never use this configuration over plain HTTP.

`prepare_hosted_archive.py` creates a private archive containing a consistent SQLite snapshot, original source images, uploads and segmentation output. Store and transfer it privately, never in a Git repository. The archive must be restored onto the persistent disk with the application stopped so an active database is not replaced. Back up the disk independently. Model weights download on first hosted segmentation, and hosted inference memory requirements have not been measured; the cheapest plan may not be sufficient. The deployment and authentication have not yet been tested. No hosted version should be used for private photos until the build, access controls, migration and upload/segmentation workflow have been verified.
