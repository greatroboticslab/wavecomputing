# `labels/`: one material list for every laser dataset

Each dataset keeps its own labels. [`materials.py`](materials.py) maps them to one shared list of materials, the union
of the 3DAI scanner's classes and the robotic arm's materials, so neither dataset has to be relabeled.

| File | Contents |
|---|---|
| [`materials.py`](materials.py) | the shared list (`CANON`), the written-out mappings (`SYNONYMS`), corrections of single labels with their reasons (`OVERRIDES`), labels the photos cannot settle (`REVIEW`), and the script that writes the tables below |
| [`out/3dai_objects.csv`](out/3dai_objects.csv) | one row per scanner object: raw and normalized material, details (subclass), surface, transparency |
| [`out/arm_classes.csv`](out/arm_classes.csv) | one row per arm class: the arm labels each laser spot with an object and the material of the part the spot lands on, so an object can have two classes (scissors: plastic handle, metal blade) |

Shared list: plastic, metal, paper, cardboard, foam, rubber, silicone, leather, fabric, wood, stone, fiberglass,
composite. Spelling is normalized (case, spaces); the only other mappings are "plastic tape" to plastic (detail tape),
"composite plastic" to composite (detail plastic) and the scanner's "fiberglass cloth press" to fiberglass (detail cloth
press). One override: the arm's cardboard shipping box body, labeled composite, is mapped to cardboard (the photos show
corrugated cardboard under those spots; to confirm with the arm's author). Two labels were confirmed by looking at the
objects (18 November 2026): the insulated water bottle and the stapler are painted metal, matte and glossy paint
respectively; they stay metal, with detail "painted" and the paint's surface recorded, since the laser reads the
surface.

Rerun after either dataset changes:

```bash
python -P labels/materials.py --tdai data/3dai/6556bec6898e/dataset/laser_features.csv \
    --arm <roboticsarmAI checkout>/dataset/data.yaml --out labels/out
```

A raw material the list does not know stops the script with a message, so a new label is mapped on purpose, never
silently.
