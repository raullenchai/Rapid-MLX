import type { ModelEntry } from '@/api/types';

/**
 * Product-facing descriptions for the audio catalog.
 *
 * Ported from `rapid-mac`'s `AudioViewModel.transcriptionDetails`. The engine's
 * alias list is technical (`whisper-large-v3-turbo`, `qwen3-tts-4bit`); this
 * layer answers the question a user actually has — which one fits my language
 * and my speed/quality tradeoff. The fallback stays useful so a newly added
 * engine alias never lands in the picker as an unexplained name.
 */

export interface ModelDetails {
  displayName: string;
  badge: string;
  summary: string;
  recommended: boolean;
}

const STT_DETAILS: Record<string, ModelDetails> = {
  'whisper-large-v3': {
    displayName: 'Whisper Large v3',
    badge: 'best quality',
    summary: 'Highest-accuracy Whisper model. Supports 99+ languages and difficult accents.',
    recommended: false,
  },
  'whisper-large-v3-turbo': {
    displayName: 'Whisper Large v3 Turbo',
    badge: 'balanced',
    summary: 'Near Large v3 accuracy, much faster. Supports 99+ languages.',
    recommended: true,
  },
  'whisper-medium': {
    displayName: 'Whisper Medium',
    badge: 'multilingual',
    summary: 'Good multilingual accuracy with lower memory use than the Large models.',
    recommended: false,
  },
  'whisper-small': {
    displayName: 'Whisper Small',
    badge: 'fast',
    summary: 'Fast, lightweight multilingual transcription for everyday dictation.',
    recommended: false,
  },
  'whisper-base': {
    displayName: 'Whisper Base',
    badge: 'low memory',
    summary: 'A compact multilingual model for older Macs. Faster, with lower accuracy.',
    recommended: false,
  },
  'whisper-tiny': {
    displayName: 'Whisper Tiny',
    badge: 'smallest',
    summary: 'The smallest Whisper. Useful when disk or memory is tight.',
    recommended: false,
  },
  'parakeet-v3': {
    displayName: 'Parakeet TDT v3',
    badge: 'English',
    summary: 'Fast English transcription with automatic punctuation.',
    recommended: false,
  },
  parakeet: {
    displayName: 'Parakeet TDT v2',
    badge: 'English',
    summary: 'English-only transcription. Very fast, with strong punctuation.',
    recommended: false,
  },
  'qwen3-asr': {
    displayName: 'Qwen3 ASR 1.7B',
    badge: 'code-switching',
    summary: 'Strong Chinese/English code-switching, punctuation and vocabulary hints.',
    recommended: false,
  },
  'qwen3-asr-0.6b': {
    displayName: 'Qwen3 ASR 0.6B',
    badge: 'fast',
    summary: 'A smaller, faster Chinese and English model with a modest accuracy tradeoff.',
    recommended: false,
  },
  sensevoice: {
    displayName: 'SenseVoice Small',
    badge: 'Asian languages',
    summary: 'Fast Chinese, Cantonese, Japanese, Korean and English with sound-event tags.',
    recommended: false,
  },
};

const TTS_DETAILS: Record<string, ModelDetails> = {
  kokoro: {
    displayName: 'Kokoro 82M',
    badge: 'recommended',
    summary: '54 built-in voices across nine languages. Small, fast, and no reference clip.',
    recommended: true,
  },
  'kokoro-4bit': {
    displayName: 'Kokoro 82M (4-bit)',
    badge: 'smallest',
    summary: 'Quantised Kokoro — a smaller download for a mild quality drop.',
    recommended: false,
  },
  'kokoro-8bit': {
    displayName: 'Kokoro 82M (8-bit)',
    badge: 'compact',
    summary: 'Quantised Kokoro at 8-bit. Close to bf16 quality, roughly half the size.',
    recommended: false,
  },
  'qwen3-tts': {
    displayName: 'Qwen3 CustomVoice',
    badge: 'named speakers',
    summary: 'Reference-free speech with named speakers. Strong Chinese and English.',
    recommended: false,
  },
  'qwen3-tts-voicedesign': {
    displayName: 'Qwen3 VoiceDesign',
    badge: 'describable',
    summary: 'Qwen3 TTS tuned for voices described in words rather than picked from a list.',
    recommended: false,
  },
  chatterbox: {
    displayName: 'Chatterbox Turbo',
    badge: 'expressive',
    summary: 'Multilingual and expressive. Clones a voice from a reference clip.',
    recommended: false,
  },
  vibevoice: {
    displayName: 'VibeVoice Realtime',
    badge: 'realtime',
    summary: 'A 0.5B model built for low-latency streaming speech.',
    recommended: false,
  },
  voxcpm: {
    displayName: 'VoxCPM 1.5',
    badge: 'natural',
    summary: 'Natural prosody on long passages, at a higher cost per second of audio.',
    recommended: false,
  },
  dia: {
    displayName: 'Dia 1.6B',
    badge: 'dialogue',
    summary: 'Tuned for multi-speaker dialogue rather than single-voice narration.',
    recommended: false,
  },
  indextts: {
    displayName: 'IndexTTS 1.5',
    badge: 'cloning',
    summary: 'Voice cloning from a reference clip, with Chinese and English support.',
    recommended: false,
  },
  'f5-tts-zh': {
    displayName: 'F5-TTS (Chinese)',
    badge: 'cloning',
    summary: 'Chinese voice cloning. Needs a reference clip and its transcript.',
    recommended: false,
  },
};

/**
 * The description for one alias.
 *
 * Quantisation suffixes are stripped before the lookup so `kokoro-82m-4bit`
 * and `kokoro-4bit` resolve to the same entry — the registry publishes half a
 * dozen spellings of every checkpoint and tabulating them all is how one gets
 * missed.
 */
export function modelDetails(entry: ModelEntry): ModelDetails {
  const table = entry.audio_kind === 'tts' ? TTS_DETAILS : STT_DETAILS;
  const alias = entry.alias.toLowerCase();
  const found = table[alias] ?? table[canonicalAlias(alias)];
  if (found) return found;

  const family = entry.family?.replace(/_/g, ' ') ?? 'Speech';
  return {
    displayName: entry.alias,
    badge: family,
    summary:
      entry.audio_kind === 'tts'
        ? 'Local text-to-speech model. Runs offline after its first download.'
        : 'Local speech-to-text model. Runs offline after its first download.',
    recommended: false,
  };
}

/** `kokoro-82m-4bit` -> `kokoro-4bit`, `parakeet-tdt-0.6b-v2` -> `parakeet`. */
function canonicalAlias(alias: string): string {
  if (alias.startsWith('kokoro')) {
    const quant = /-(4bit|8bit)$/.exec(alias);
    return quant ? `kokoro-${quant[1]}` : 'kokoro';
  }
  if (alias.startsWith('parakeet')) return alias.includes('v3') ? 'parakeet-v3' : 'parakeet';
  if (alias === 'whisper' || alias === 'whisper-1') return 'whisper-large-v3';
  if (alias === 'qwen3-asr-1.7b') return 'qwen3-asr';
  if (alias === 'sensevoice-small') return 'sensevoice';
  if (alias.startsWith('qwen3-tts-voicedesign')) return 'qwen3-tts-voicedesign';
  if (alias.startsWith('qwen3-tts')) return 'qwen3-tts';
  if (alias.startsWith('chatterbox')) return 'chatterbox';
  if (alias.startsWith('vibevoice')) return 'vibevoice';
  if (alias.startsWith('indextts')) return 'indextts';
  return alias;
}

/**
 * The audio rows of one kind, one row per checkpoint.
 *
 * The registry exposes compatibility aliases for API and CLI callers
 * (`whisper`, `whisper-1` and `whisper-large-v3` are one repo), but a visual
 * picker showing the same weights three times is just noise. Grouped by
 * `hf_path`, keeping the alias a person would recognise.
 */
export function audioModels(models: ModelEntry[], kind: 'tts' | 'stt'): ModelEntry[] {
  const byRepo = new Map<string, ModelEntry>();
  for (const entry of models) {
    if (entry.kind !== 'audio' || entry.audio_kind !== kind) continue;
    const key = entry.hf_path.toLowerCase();
    const current = byRepo.get(key);
    if (!current || aliasRank(entry.alias) < aliasRank(current.alias)) byRepo.set(key, entry);
  }

  return [...byRepo.values()].sort((left, right) => {
    // Recommended first, then downloaded, then alphabetical: on a catalog
    // this size the ones already on disk are what the user came for.
    const leftPick = modelDetails(left).recommended;
    const rightPick = modelDetails(right).recommended;
    if (leftPick !== rightPick) return leftPick ? -1 : 1;
    if (left.cached !== right.cached) return left.cached ? -1 : 1;
    return left.alias.localeCompare(right.alias);
  });
}

/** Which spelling of a repeated checkpoint the picker shows. Lower wins. */
function aliasRank(alias: string): number {
  switch (alias.toLowerCase()) {
    // The product names.
    case 'whisper-large-v3':
    case 'parakeet':
    case 'parakeet-v3':
    case 'qwen3-asr':
    case 'sensevoice':
    case 'kokoro':
    case 'qwen3-tts':
      return 0;
    // The API-compatibility spellings.
    case 'whisper':
    case 'whisper-1':
    case 'parakeet-tdt-0.6b':
    case 'parakeet-tdt-0.6b-v2':
    case 'parakeet-tdt-0.6b-v3':
    case 'qwen3-asr-1.7b':
    case 'sensevoice-small':
    case 'kokoro-82m':
    case 'kokoro-82m-bf16':
    case 'qwen3-tts-customvoice':
      return 2;
    default:
      return 1;
  }
}

/**
 * Which alias to start on.
 *
 * A DOWNLOADED model wins over any preference: the lane can use it right now,
 * and defaulting to something absent means the panel opens onto a model the
 * user cannot run. Only when nothing is cached does the preference list apply.
 */
export function preferredAlias(
  entries: ModelEntry[],
  preferred: string[],
): string | null {
  const cached = entries.find((entry) => entry.cached);
  if (cached) return cached.alias;
  for (const alias of preferred) {
    if (entries.some((entry) => entry.alias === alias)) return alias;
  }
  return entries[0]?.alias ?? null;
}

export const PREFERRED_STT = ['whisper-large-v3-turbo', 'whisper-small', 'whisper-large-v3'];
export const PREFERRED_TTS = ['kokoro', 'kokoro-8bit', 'qwen3-tts'];

/**
 * What a Kokoro voice id means.
 *
 * The ids are systematic — `af_heart` is American English, female — so this
 * decodes the prefix rather than tabulating 54 names. Qwen3's named speakers
 * are not systematic and are listed individually.
 */
const NAMED_VOICES: Record<string, string> = {
  vivian: 'Chinese · Female',
  serena: 'Chinese · Female',
  uncle_fu: 'Chinese · Male',
  dylan: 'Chinese · Beijing · Male',
  eric: 'Chinese · Sichuan · Male',
  ryan: 'English · Male',
  aiden: 'English · Male',
  ono_anna: 'Japanese · Female',
  sohee: 'Korean · Female',
};

const VOICE_LANGUAGES: Record<string, string> = {
  a: 'American English',
  b: 'British English',
  e: 'Spanish',
  f: 'French',
  h: 'Hindi',
  i: 'Italian',
  j: 'Japanese',
  p: 'Portuguese',
  z: 'Chinese',
};

export function voiceDetails(voice: string): string {
  const named = NAMED_VOICES[voice.toLowerCase()];
  if (named) return named;

  const code = /^([abefhijpz])([fm])_/.exec(voice.toLowerCase());
  if (code) {
    const language = VOICE_LANGUAGES[code[1] ?? ''];
    if (language) return `${language} · ${code[2] === 'f' ? 'Female' : 'Male'}`;
  }
  return 'Multilingual';
}

/** What a preview says, in the voice's own language — an English sample from
 *  a Japanese voice demonstrates the wrong thing. */
export function previewText(voice: string): string {
  const lower = voice.toLowerCase();
  if (/^z[fm]_/.test(lower) || ['vivian', 'serena', 'uncle_fu', 'dylan', 'eric'].includes(lower)) {
    return '你好，这是我的声音，很高兴认识你。';
  }
  if (/^j[fm]_/.test(lower) || lower === 'ono_anna') return 'こんにちは、私の声を聞いてください。';
  if (lower === 'sohee') return '안녕하세요, 제 목소리를 들어 보세요.';
  if (/^e[fm]_/.test(lower)) return 'Hola, esta es una muestra de mi voz.';
  if (/^f[fm]_/.test(lower)) return 'Bonjour, voici un aperçu de ma voix.';
  return 'Hello, this is a preview of my voice.';
}
