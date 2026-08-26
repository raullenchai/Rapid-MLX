#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

# Resolve an explicit GitHub tag ref to a commit, following at most ten
# annotated tag objects. Callers provide GH_BIN and GITHUB_REPOSITORY.
resolve_github_tag_commit() {
  local tag="$1"
  local object_line object_type object_sha depth=0
  object_line=$(
    "${GH_BIN:-gh}" api \
      "repos/${GITHUB_REPOSITORY:?}/git/ref/tags/$tag" \
      --jq '.object | [.type, .sha] | @tsv'
  ) || return 1
  IFS=$'\t' read -r object_type object_sha <<<"$object_line"
  while true; do
    case "$object_type" in
      commit)
        [[ "$object_sha" =~ ^[0-9a-f]{40}$ ]] || return 1
        printf '%s\n' "$object_sha"
        return 0
        ;;
      tag)
        depth=$((depth + 1))
        [[ "$depth" -le 10 ]] || return 1
        object_line=$(
          "${GH_BIN:-gh}" api \
            "repos/${GITHUB_REPOSITORY}/git/tags/$object_sha" \
            --jq '.object | [.type, .sha] | @tsv'
        ) || return 1
        IFS=$'\t' read -r object_type object_sha <<<"$object_line"
        ;;
      *) return 1 ;;
    esac
  done
}
