<!-- Draft for the next release. Rename in the version-bump PR. -->

- Add the pinned `ornith-1.5-122b-a10b-4bit` alias with a 96 GB memory floor,
  hybrid MoE metadata, named XML tool parsing, and documented qualification.
- Add native LFM3 text support and the pinned `lfm3-1.7b-8bit` alias,
  with thinking, XML tool calls, and hybrid prefix-cache support.

- Add native image support for LFM3-VL and the pinned `lfm3-vl-2b-8bit` alias.
- Fix vision requests with ordinary-sized photos sent as `image_url` data URLs
  being rejected as `context_length_exceeded` on models whose chat template
  serializes unknown content parts as text (for example LFM3-VL).
