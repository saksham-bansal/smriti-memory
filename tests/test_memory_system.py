"""
Unit and Integration Test Suite for Smriti Agent Memory Across Sessions.

Tests:
  1. Hard Requirement 1: Persistence across independent agent lifecycles / restarts
  2. Hard Requirement 2: Claim-level provenance hydration and fabricated-ID stripping
  3. Relative-to-absolute temporal normalization ('yesterday', '4 years ago', 'last Friday')
  4. Bi-temporal state supersession (`ACTIVE` -> `SUPERSEDED`) & Dream synthesis
  5. Layer 3 Linked Wiki & Layer 4 `global.md` compilation with strict turn provenance
  6. Adversarial / False-premise abstention guard
"""
from pathlib import Path

from smriti.agent.responder import SmritiMemoryAgent
from smriti.config import SmritiConfig
from smriti.ingestion.temporal import TemporalNormalizer
from smriti.models import ClaimProvenance, ClaimStatus, SessionTranscript, Turn


def _build_two_sessions(conv_id: str = "test_conv"):
    s1 = SessionTranscript(
        session_id="session_1",
        conversation_id=conv_id,
        timestamp="1:56 pm on 8 May, 2023",
        turns=[
            Turn(
                turn_id="D1:1",
                session_id="session_1",
                conversation_id=conv_id,
                speaker="Caroline",
                text="I currently live in Seattle and I follow a vegetarian diet.",
                timestamp="1:56 pm on 8 May, 2023",
                turn_index=1,
            ),
            Turn(
                turn_id="D1:2",
                session_id="session_1",
                conversation_id=conv_id,
                speaker="Melanie",
                text="I went to an LGBTQ support group yesterday and painted a sunrise.",
                timestamp="1:56 pm on 8 May, 2023",
                turn_index=2,
            ),
            Turn(
                turn_id="D1:3",
                session_id="session_1",
                conversation_id=conv_id,
                speaker="Caroline",
                text="I moved from Sweden 4 years ago to pursue counseling.",
                timestamp="1:56 pm on 8 May, 2023",
                turn_index=3,
            ),
        ],
    )
    s2 = SessionTranscript(
        session_id="session_2",
        conversation_id=conv_id,
        timestamp="4:00 pm on 15 June, 2023",
        turns=[
            Turn(
                turn_id="D2:1",
                session_id="session_2",
                conversation_id=conv_id,
                speaker="Caroline",
                text="I moved to Portland last Friday and I switched to a vegan diet!",
                timestamp="4:00 pm on 15 June, 2023",
                turn_index=1,
            ),
            Turn(
                turn_id="D2:2",
                session_id="session_2",
                conversation_id=conv_id,
                speaker="Melanie",
                text="That is exciting! I signed up for a pottery class next month.",
                timestamp="4:00 pm on 15 June, 2023",
                turn_index=2,
            ),
        ],
    )
    return s1, s2


def test_persistence_across_restarts(tmp_path: Path):
    """Hard Requirement 1: Memory must persist across separate agent instances / restarts."""
    db_file = str(tmp_path / "persistent_memory.db")
    s1, s2 = _build_two_sessions("conv_persist")

    # Run 1: Ingest Session 1 and close database connection completely
    agent1 = SmritiMemoryAgent(SmritiConfig(db_path=db_file))
    agent1.ingest_session(s1, run_dream=False)
    agent1.close()

    # Run 2: New instance, ingest Session 2 + run offline Dream consolidation, then close
    agent2 = SmritiMemoryAgent(SmritiConfig(db_path=db_file))
    agent2.ingest_session(s2, run_dream=True)
    agent2.close()

    # Run 3: Third instance queries persisted memory across both sessions
    agent3 = SmritiMemoryAgent(SmritiConfig(db_path=db_file))
    try:
        all_turns = agent3.store.get_all_turns("conv_persist")
        assert len(all_turns) == 5

        wikis = agent3.store.get_wiki_pages("conv_persist")
        assert len(wikis) >= 2

        gh = agent3.store.get_global_header("conv_persist")
        assert gh is not None
        assert "Caroline" in gh.header_markdown
        assert "Portland" in gh.header_markdown

        resp = agent3.answer_query("conv_persist", "Where does Caroline live currently?")
        assert not resp.abstained
        assert "Portland" in resp.answer
        assert "D2:1" in resp.all_cited_turn_ids
    finally:
        agent3.close()


def test_claim_level_provenance_and_fabrication_stripping(tmp_path: Path):
    """Hard Requirement 2: Every claim must trace to verified Layer 1 turns; fabricated IDs stripped."""
    db_file = str(tmp_path / "provenance.db")
    s1, _ = _build_two_sessions("conv_prov")

    agent = SmritiMemoryAgent(SmritiConfig(db_path=db_file))
    try:
        agent.ingest_session(s1, run_dream=True)

        # Test programmatic stripping of hallucinated/non-existent turn IDs
        raw_claims = [
            ClaimProvenance(
                claim_text="Melanie attended a support group on 7 May 2023.",
                source_turn_ids=["HALLUCINATED_ID_99", "D1:2", "NON_EXISTENT_42"],
            )
        ]
        verified = agent.verifier.verify_and_hydrate_claims("conv_prov", raw_claims, [])
        assert len(verified) == 1
        assert verified[0].source_turn_ids == ["D1:2"]
        assert verified[0].source_session_ids == ["session_1"]
        assert "LGBTQ support group" in verified[0].supporting_quotes[0]
        assert verified[0].verified is True
    finally:
        agent.close()


def test_temporal_anchoring():
    """Verify relative temporal expressions are anchored to session timestamps."""
    anchors, primary = TemporalNormalizer.extract_temporal_anchors(
        "I went to an LGBTQ support group yesterday.", "1:56 pm on 8 May, 2023"
    )
    assert primary == "7 May 2023"
    assert any("7 May 2023" in a for a in anchors)

    anchors_yr, primary_yr = TemporalNormalizer.extract_temporal_anchors(
        "I moved from Sweden 4 years ago.", "1:56 pm on 8 May, 2023"
    )
    assert primary_yr == "2019"
    assert any("2019" in a for a in anchors_yr)

    anchors_fri, primary_fri = TemporalNormalizer.extract_temporal_anchors(
        "I moved to Portland last Friday.", "4:00 pm on 15 June, 2023"
    )
    assert primary_fri == "9 June 2023"


def test_bitemporal_supersession_and_dream_wiki(tmp_path: Path):
    """Verify contradictory state updates mark old claims SUPERSEDED while preserving history."""
    db_file = str(tmp_path / "bitemporal.db")
    s1, s2 = _build_two_sessions("conv_bi")

    agent = SmritiMemoryAgent(SmritiConfig(db_path=db_file))
    try:
        agent.ingest_session(s1)
        agent.ingest_session(s2)
        stats = agent.dream("conv_bi")
        assert stats["superseded_count"] >= 2
        assert stats["synthesis_claims"] >= 2

        loc_claims = agent.store.get_claims(
            "conv_bi", subject="Caroline", predicate="lives_in"
        )
        superseded = [c for c in loc_claims if c.status == ClaimStatus.SUPERSEDED.value]
        active = [c for c in loc_claims if c.status == ClaimStatus.ACTIVE.value]

        assert any("Seattle" in c.object_value for c in superseded)
        assert any("Portland" in c.object_value for c in active)
        assert superseded[0].valid_until == "4:00 pm on 15 June, 2023"
        assert superseded[0].superseded_by is not None

        # Historical query should still recall Seattle with D1:1 provenance
        hist_resp = agent.answer_query(
            "conv_bi", "Where did Caroline live previously before moving?"
        )
        assert "Seattle" in hist_resp.answer
        assert "D1:1" in hist_resp.all_cited_turn_ids
    finally:
        agent.close()


def test_adversarial_and_unanswerable_abstention(tmp_path: Path):
    """Verify false-premise speaker swaps and unmentioned topics trigger abstention."""
    db_file = str(tmp_path / "adversarial.db")
    s1, s2 = _build_two_sessions("conv_adv")

    agent = SmritiMemoryAgent(SmritiConfig(db_path=db_file))
    try:
        agent.ingest_session(s1)
        agent.ingest_session(s2, run_dream=True)

        # Unmentioned topic
        r_unmentioned = agent.answer_query(
            "conv_adv", "What cryptocurrency did Caroline invest in?"
        )
        assert r_unmentioned.abstained is True

        # Speaker-swap adversarial trap (Melanie painted a sunrise in D1:2, not Caroline)
        r_swap = agent.answer_query(
            "conv_adv", "When did Caroline paint a sunrise at the LGBTQ support group?"
        )
        assert r_swap.abstained is True
        assert "Melanie" in r_swap.answer
    finally:
        agent.close()
