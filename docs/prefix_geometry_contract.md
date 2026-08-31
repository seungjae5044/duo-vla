# Fixed prefix geometry contract

DiffusionGemma's grouped MoE kernels can depend on the physical batch geometry. Duo-VLA therefore treats the padded
multimodal prefix width as an authenticated experiment input, rather than a dataloader implementation detail. The
generic `duo-vla-prefix-geometry-v1` artifact records:

- separate model and processor repository IDs, pinned 40-character revisions, authenticated snapshot-tree hashes,
  content-inventory hashes, file counts, and byte counts;
- ordered raw RGB camera names and exact HWC shapes;
- the sorted, unique, exact-UTF-8 instruction inventory, its SHA-256 and count, and an individual SHA-256 and measured
  valid-prefix length for every instruction;
- the derived length histogram and maximum, tokenizer padding side, and one explicit fixed physical prefix width.

The artifact does not choose a width for LIBERO or CALVIN. A benchmark-specific artifact may be created only after its
complete training and evaluation instruction inventory has been authenticated and scanned. The chosen width must be at
least one position greater than the measured maximum. This sentinel padding keeps every row on the explicit attention-
mask path instead of allowing an all-valid batch to elide its SDPA mask.

## Measurement and runtime rule

`create_prefix_geometry_contract` measures each instruction twice with all-zero `uint8` image probes at the declared
camera geometries. The unbounded call uses `padding=False, truncation=False`. The fixed call uses exactly:

```python
processor_kwargs = {
    "padding": "max_length",
    "max_length": fixed_physical_prefix_width,
    "truncation": False,
}
```

The two valid-token counts must agree. `input_ids`, `attention_mask`, and `mm_token_type_ids` must be prefix-aligned
rank-two tensors with the same `[B, P]` shape; token IDs are integral and the attention mask is binary and contiguous.
`pixel_values` and `image_position_ids` must both have first-axis size `B * C`, where `C` is the authenticated ordered
camera count. These five required output fields and the image-axis meaning are part of the artifact schema. The fixed
tensor width must equal the contract width exactly. This last check is important: with `truncation=False`, Transformers
can return a tensor wider than `max_length` for an overlength sample. Duo-VLA rejects that tensor instead of silently
truncating it.

Training and serving call `apply_fixed_prefix_chat_template`, passing the fixed physical batch size as
`expected_batch_size` and the verified ordered-camera count as `images_per_prefix`. They additionally require every
runtime instruction to occur in the authenticated inventory and compare its observed valid-token count with the
recorded per-instruction length. LIBERO pins B=8, two ordered `256x256x3` cameras, 40 instructions, a maximum valid
length of 544, and `P=545`. Its `libero-v2.json` semantic SHA-256 is
`cc907e22ccd5ae704767edba606233dede39989a119ac544764b47aaa4fbe634`.

CALVIN pins B=8, ordered `200x200x3` static and `84x84x3` gripper cameras, a maximum valid length of 537, and `P=538`.
Its `calvin-abc-to-d-v1.json` semantic SHA-256 is
`edaef86df702e34c9be6c9103e4f3c9ccc4022df8def46df7d1f00084d0831c9`.

## Authentication and file handling

Create and publish an immutable artifact as follows:

```python
contract = create_prefix_geometry_contract(
    processor,
    model_identity=model_identity,
    processor_identity=processor_identity,
    ordered_cameras=ordered_cameras,
    instructions=instructions,
    fixed_physical_prefix_width=chosen_width,
    padding_side=processor.tokenizer.padding_side,
)
content_sha256 = save_prefix_geometry_contract(output_path, contract)
```

`save_prefix_geometry_contract` exclusively publishes one canonical, sorted JSON representation and refuses to
overwrite an existing path. It opens every parent directory and the file with no-follow semantics.

Every consumer must obtain `content_sha256` from a separately authenticated run configuration or checkpoint and pass
it to `load_prefix_geometry_contract`. A self-hash alone cannot prevent an attacker from replacing both an artifact and
its self-hash; the externally pinned digest closes that substitution gap. Loading also rejects duplicate JSON keys,
non-canonical serialization, unknown schema fields, symlinked path components, inventory or histogram inconsistencies,
and any expected model, processor, camera, instruction, or width mismatch.
