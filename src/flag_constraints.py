from __future__ import annotations

###############################################################################
# Configurable constraints over compiler-flag combinations.                   #
#                                                                             #
# A config can declare, once, which flags are mutually exclusive and which    #
# flags are already implied by another active flag.  Search methods call into #
# this module before building a candidate so they never test a combination    #
# that cancels itself out or is redundant.                                    #
#                                                                             #
# Everything here is pure string logic: a "flag" is the exact string entry of #
# ``compiler_flags`` or ``compiler_flag_pool`` (or a rendered compiler param). #
###############################################################################

import dataclasses
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

###############################################################################
# Type helpers                                                                #
###############################################################################

_ALLOWED_KEYS = {"exclusive", "implies"}


def _as_flag_list(raw: Any, *, where: str) -> List[str]:
    """Validate a JSON value that must be a list of non-empty flag strings."""
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{where} must be a list of flag strings")
    flags: List[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{where} entries must be non-empty strings")
        flags.append(item)
    return flags


###############################################################################
# FlagConstraints                                                             #
###############################################################################

@dataclasses.dataclass(frozen=True)
class FlagConstraints:
    """
    Declarative rules over flag combinations.

    ``exclusive_groups``
        Each group lists flags of which **at most one** may be active at a time
        (for example ``{"-funroll-loops", "-fno-unroll-loops"}``).

    ``implies``
        ``source -> [implied, ...]``.  When ``source`` is active, every implied
        flag is redundant and is dropped.  The relation is applied transitively.
        Implied flags are **not** treated as active, so overriding tests such as
        ``-O3 -fno-unroll-loops`` stay legal.
    """

    exclusive_groups: Tuple[FrozenSet[str], ...] = ()
    implies: Mapping[str, FrozenSet[str]] = dataclasses.field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, raw: Optional[Mapping[str, Any]]) -> "FlagConstraints":
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError("flag_constraints must be an object")
        unknown = [key for key in raw if key not in _ALLOWED_KEYS]
        if unknown:
            raise ValueError(
                f"flag_constraints has unknown keys: {sorted(unknown)}; "
                f"allowed: {sorted(_ALLOWED_KEYS)}"
            )

        groups: List[FrozenSet[str]] = []
        for index, raw_group in enumerate(raw.get("exclusive", []) or []):
            members = _as_flag_list(raw_group, where=f"flag_constraints.exclusive[{index}]")
            if len(members) < 2:
                raise ValueError(
                    f"flag_constraints.exclusive[{index}] needs at least two flags"
                )
            if len(set(members)) != len(members):
                raise ValueError(
                    f"flag_constraints.exclusive[{index}] contains duplicate flags"
                )
            groups.append(frozenset(members))

        implies_raw = raw.get("implies", {}) or {}
        if not isinstance(implies_raw, Mapping):
            raise ValueError("flag_constraints.implies must be an object")
        implies: Dict[str, FrozenSet[str]] = {}
        for key, values in implies_raw.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("flag_constraints.implies keys must be non-empty strings")
            implied = frozenset(
                v for v in _as_flag_list(values, where=f"flag_constraints.implies[{key!r}]")
                if v != key
            )
            if implied:
                implies[key] = implied

        return cls(exclusive_groups=tuple(groups), implies=implies)

    # ------------------------------------------------------------------
    # Introspection / serialisation
    # ------------------------------------------------------------------
    @property
    def is_empty(self) -> bool:
        return not self.exclusive_groups and not self.implies

    def as_fingerprint(self) -> Dict[str, Any]:
        """Canonical, order-insensitive form (used by the resume fingerprint)."""
        return {
            "exclusive": sorted(sorted(group) for group in self.exclusive_groups),
            "implies": {key: sorted(values) for key, values in sorted(self.implies.items())},
        }

    def validate_against(self, universe: Iterable[str]) -> List[str]:
        """
        Return human-readable warnings for flags referenced here that are absent
        from ``universe`` (the configured compiler_flags/compiler_flag_pool).
        """
        known = set(universe)
        warnings: List[str] = []
        for index, group in enumerate(self.exclusive_groups):
            unknown = sorted(flag for flag in group if flag not in known)
            if unknown:
                warnings.append(
                    f"flag_constraints.exclusive[{index}] references flags outside "
                    f"compiler_flags/compiler_flag_pool: {unknown}"
                )
        for key, values in self.implies.items():
            if key not in known:
                warnings.append(
                    f"flag_constraints.implies key {key!r} is not in "
                    "compiler_flags/compiler_flag_pool"
                )
            unknown = sorted(value for value in values if value not in known)
            if unknown:
                warnings.append(
                    f"flag_constraints.implies[{key!r}] references flags outside "
                    f"compiler_flags/compiler_flag_pool: {unknown}"
                )
        return warnings

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def implied_flags(self, active: Iterable[str]) -> FrozenSet[str]:
        """Transitive closure of everything implied by the active flags."""
        seen: set[str] = set()
        frontier = [flag for flag in set(active) if flag in self.implies]
        while frontier:
            current = frontier.pop()
            for implied in self.implies.get(current, ()):  # type: ignore[arg-type]
                if implied not in seen:
                    seen.add(implied)
                    if implied in self.implies:
                        frontier.append(implied)
        return frozenset(seen)

    def conflicts(self, flag: str, active: Iterable[str]) -> bool:
        """True if activating ``flag`` alongside ``active`` breaks an exclusive group."""
        active_set = set(active)
        for group in self.exclusive_groups:
            if flag in group and (group - {flag}) & active_set:
                return True
        return False

    def is_consistent(self, flags: Iterable[str]) -> bool:
        """True if no exclusive group has more than one active member."""
        present = set(flags)
        for group in self.exclusive_groups:
            if len(group & present) > 1:
                return False
        return True

    def can_add(self, flag: str, active: Iterable[str]) -> bool:
        """True if ``flag`` may be added to ``active`` without redundancy/conflict."""
        if self.is_empty:
            return True
        active_set = set(active)
        if flag in active_set:
            return False
        if self.conflicts(flag, active_set):
            return False
        # Redundant: already implied by the current context.
        if flag in self.implied_flags(active_set):
            return False
        # Adding it would make an already-active flag redundant.
        if self.implied_flags([flag]) & active_set:
            return False
        return True

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------
    def prune_pool(self, pool: Sequence[str], active: Iterable[str]) -> List[str]:
        """Drop pool entries that are redundant given ``active``."""
        redundant = self.implied_flags(active)
        return [flag for flag in pool if flag not in redundant]

    def expand_pool(self, pool: Sequence[str]) -> List[str]:
        """
        Return ``pool`` plus any exclusive-group member it does not already list.

        Members of an ``exclusive`` group are implicitly part of the flag space,
        so they do not have to be repeated in ``compiler_flag_pool``. This does
        **not** apply to ``implies`` values, which must be listed explicitly.
        """
        seen: set = set()
        out: List[str] = []
        for flag in pool:
            if flag not in seen:
                seen.add(flag)
                out.append(flag)
        for group in self.exclusive_groups:
            for flag in sorted(group):
                if flag not in seen:
                    seen.add(flag)
                    out.append(flag)
        return out

    def effective_signature(self, flags: Sequence[str]) -> Tuple[str, ...]:
        """
        Order-preserving canonical form of a flag list with implied flags removed.

        Two flag lists with the same signature describe the same effective
        compiler configuration, so a search may reuse the earlier result.

        Only the transitive ``implies`` relation is applied. Exclusive groups are
        deliberately **not** resolved here: silently dropping one member of a
        conflicting pair would collapse a configuration the caller asked for into
        a different one, which could make the search skip a genuinely distinct
        node. Conflicting combinations are prevented where they are generated.
        """
        ordered = list(dict.fromkeys(flags))
        if not self.implies:
            return tuple(ordered)
        redundant = self.implied_flags(ordered)
        return tuple(flag for flag in ordered if flag not in redundant)

    def sanitize(self, flags: Sequence[str]) -> List[str]:
        """
        Remove redundant flags and surplus members of exclusive groups.

        Earlier flags win: the input is expected to be ordered
        ``[base, variant, params..., pool...]``.
        """
        ordered = list(dict.fromkeys(flags))
        if self.is_empty:
            return ordered

        redundant = set(self.implied_flags(ordered))
        kept: List[str] = []
        context: set[str] = set()
        for flag in ordered:
            if flag in redundant:
                continue
            if self.conflicts(flag, context):
                continue
            kept.append(flag)
            context.add(flag)
        return kept

    def sanitize_pool(
        self,
        active: Sequence[str],
        pool: Sequence[str],
    ) -> List[str]:
        """Return the subset of ``pool`` that stays legal alongside ``active``."""
        if self.is_empty:
            return list(pool)
        cleaned = self.sanitize(list(active) + list(pool))
        active_set = set(active)
        return [flag for flag in cleaned if flag not in active_set]

    def can_add_exclusive(self, flag: str, active: Iterable[str]) -> bool:
        """
        True if ``flag`` may be added without breaking an exclusive group.

        Unlike :meth:`can_add` this ignores ``implies``. An implied flag is a
        legal (if redundant) node whose result is reused through equivalence
        detection rather than removed from the search space.
        """
        if flag in set(active):
            return False
        return not self.conflicts(flag, active)

    def sanitize_exclusive(self, flags: Sequence[str]) -> List[str]:
        """
        Keep at most one member per exclusive group, leaving implied flags in
        place. Used where redundant nodes are allowed to stay traversable.
        """
        ordered = list(dict.fromkeys(flags))
        if not self.exclusive_groups:
            return ordered
        kept: List[str] = []
        context: set = set()
        for flag in ordered:
            if self.conflicts(flag, context):
                continue
            kept.append(flag)
            context.add(flag)
        return kept

    def is_allowed(self, flags: Sequence[str]) -> bool:
        """True if ``flags`` contains no conflicts and no redundancy."""
        if self.is_empty:
            return True
        ordered = list(dict.fromkeys(flags))
        return self.sanitize(ordered) == ordered

    def select_consistent(
        self,
        candidates: Sequence[str],
        active: Iterable[str] = (),
    ) -> List[str]:
        """
        Greedily keep the candidates that are legal alongside ``active``.

        Order-independent with respect to the ``implies`` relation: a candidate
        implied by another candidate (or by ``active``) is always dropped in
        favour of the implying flag.  Exclusive-group conflicts keep the first
        candidate encountered.
        """
        active_set = set(active)
        candidate_list = list(dict.fromkeys(candidates))
        if self.is_empty:
            return candidate_list

        redundant = set(self.implied_flags(active_set))
        redundant |= set(self.implied_flags(candidate_list))

        kept: List[str] = []
        context = set(active_set)
        for flag in candidate_list:
            if flag in context or flag in redundant:
                continue
            if self.conflicts(flag, context):
                continue
            kept.append(flag)
            context.add(flag)
        return kept
