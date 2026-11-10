// Vendored from site origin/main 7df4213c30fb295bfe10953399fcc95883a1aaea.
function atomicValidateModel(model, taskType) {
  let why = atomicExactObject(
    model,
    ["schema_version", "identity_strength", "pipeline_kind", "components"],
    ["schema_version", "identity_strength", "pipeline_kind", "components"],
    "model",
  );
  if (why) return why;
  if (model.schema_version !== 1 || model.identity_strength !== "unresolved") {
    return "model identity must be an unresolved v1 identity in this internal beta";
  }
  if (model.pipeline_kind !== taskType || !ATOMIC_BENCH_TASKS.has(taskType)) {
    return "model.pipeline_kind does not match the workload";
  }
  if (!Array.isArray(model.components) ||
      model.components.length < 1 || model.components.length > 32) {
    return "model.components must contain between 1 and 32 components";
  }

  const componentIds = new Set();
  let primaryCount = 0;
  let previousId = null;
  for (let index = 0; index < model.components.length; index += 1) {
    const component = model.components[index];
    const label = `model.components[${index}]`;
    why = atomicExactObject(
      component,
      ["component_id", "role", "source", "artifact", "quantization"],
      ["component_id", "role", "source", "artifact", "quantization"],
      label,
    );
    if (why) return why;
    if (typeof component.component_id !== "string" ||
        !/^[a-z0-9][a-z0-9._-]{0,63}$/.test(component.component_id)) {
      return `${label}.component_id`;
    }
    if (componentIds.has(component.component_id)) {
      return `model.components duplicate component_id ${component.component_id}`;
    }
    if (previousId !== null && previousId >= component.component_id) {
      return "model.components must be sorted by component_id";
    }
    componentIds.add(component.component_id);
    previousId = component.component_id;
    if (!ATOMIC_COMPONENT_ROLES.has(component.role)) return `${label}.role`;
    if (component.role === "primary") primaryCount += 1;

    why = atomicExactObject(
      component.source,
      ["kind", "repo_id", "subfolder", "resolved_revision"],
      ["kind", "repo_id"],
      `${label}.source`,
    );
    if (why) return why;
    if (component.source.kind !== "huggingface" ||
        !/^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/.test(component.source.repo_id)) {
      return `${label}.source must be a public Hugging Face repository id`;
    }
    // Optional provenance the client fills from the local cache. The
    // identity stays unresolved, so no manifest digest is claimed here.
    if (component.source.subfolder !== undefined &&
        !atomicSubfolder(component.source.subfolder)) {
      return `${label}.source.subfolder`;
    }
    if (component.source.resolved_revision !== undefined &&
        !(typeof component.source.resolved_revision === "string" &&
          /^[0-9a-f]{40}$/.test(component.source.resolved_revision))) {
      return `${label}.source.resolved_revision`;
    }
    why = atomicExactObject(
      component.artifact, ["format"], ["format"], `${label}.artifact`,
    );
    if (why) return why;
    if (component.artifact.format !== "mlx-safetensors") {
      return `${label}.artifact.format`;
    }
    why = atomicValidateQuantization(component.quantization, `${label}.quantization`);
    if (why) return why;
  }
  return primaryCount === 1 ? null : "model.components must contain exactly one primary role";
}

function atomicSubfolder(value) {
  if (typeof value !== "string" || !value.length || value.length > 512) return false;
  if (value.startsWith("/") || /[\\\r\n]/.test(value)) return false;
  return !value.split("/").some((segment) => segment === "..");
}

// model-identity.schema.json#/$defs/quantization, applied literally. Before
// 0.13.5 every client sent {kind: unknown, base_dtype: unknown}; that shape
// stays valid. Real facts (4-bit affine, group size 64, ...) are now
// accepted so the board can eventually say what was measured.
const ATOMIC_QUANT_KINDS = ["none", "weights", "weights_and_activations", "mixed", "unknown"];
const ATOMIC_BASE_DTYPES = ["float32", "float16", "bfloat16", "other", "unknown"];
const ATOMIC_COMPONENT_ROLES = new Set([
  "primary", "language_model", "vision_encoder", "audio_encoder", "audio_decoder",
  "vocoder", "projector", "denoiser", "vae", "text_encoder", "tokenizer",
  "processor", "adapter", "controlnet", "safety_checker", "other",
]);

function atomicValidateQuantization(quantization, label) {
  const why = atomicExactObject(
    quantization,
    ["kind", "method", "weight_bits_x2", "activation_bits_x2", "group_size", "base_dtype"],
    ["kind"],
    label,
  );
  if (why) return why;
  const q = quantization;
  if (!ATOMIC_QUANT_KINDS.includes(q.kind)) return `${label}.kind`;
  if (q.base_dtype !== undefined && !ATOMIC_BASE_DTYPES.includes(q.base_dtype)) {
    return `${label}.base_dtype`;
  }
  if (q.method !== undefined &&
      !(typeof q.method === "string" && /^[a-z0-9][a-z0-9._-]{0,63}$/.test(q.method))) {
    return `${label}.method`;
  }
  for (const field of ["weight_bits_x2", "activation_bits_x2"]) {
    if (q[field] !== undefined && !atomicInteger(q[field], 2, 64)) return `${label}.${field}`;
  }
  if (q.group_size !== undefined && !atomicInteger(q.group_size, 1, 4096)) {
    return `${label}.group_size`;
  }
  if (q.kind === "none" &&
      ["method", "weight_bits_x2", "activation_bits_x2", "group_size"].some((f) => q[f] !== undefined)) {
    return `${label}: kind none carries no quantization fields`;
  }
  if ((q.kind === "weights" || q.kind === "weights_and_activations") &&
      (q.method === undefined || q.weight_bits_x2 === undefined)) {
    return `${label}: ${q.kind} requires method and weight_bits_x2`;
  }
  if (q.kind === "weights_and_activations" && q.activation_bits_x2 === undefined) {
    return `${label}: weights_and_activations requires activation_bits_x2`;
  }
  return null;
}

