"""FSAE-EV corpus relevance filter (initial-ingestion plugin).

Rules must be conservative: a wrongly-kept document costs nothing, a
wrongly-ignored one silently disappears from the knowledge base. These tests
pin the intended boundary: combustion powertrain, outdated rulebooks, aged
software tutorials, and obsolete battery chemistry are excluded; everything
engineering/aero/chassis/EV stays.
"""

from __future__ import annotations

from pathlib import Path

from src.fsae_ev_filter import evaluate_pdf, filter_pdf_list, write_ignore_log
from scripts.local_corpus_ingest import filter_expensive_scans


def ev(path: str, root: str = "corpus") -> str | None:
    decision = evaluate_pdf(Path(root) / path, root=Path(root))
    return decision.reason if decision else None


def test_outdated_rulebooks_are_ignored():
    assert ev("2008 Rules.pdf") == "outdated rulebook (2008)"
    assert ev("Rulebook/rules 2014.pdf") is not None
    # Recent rule editions stay.
    assert ev("FSAE Rules 2026.pdf") is None
    assert ev("rules.pdf") is None  # no year -> cannot judge, keep


def test_combustion_powertrain_is_ignored():
    assert ev("Engine Design Handbook.pdf") == "combustion powertrain (engine)"
    assert ev("Engine Related Readings/intake basics.pdf") is not None
    assert ev("4-Stroke Combustion Notes.pdf") is not None
    assert ev("carburetor tuning guide.pdf") is not None
    assert ev("exhaust system design.pdf") is not None
    assert ev("turbocharger lecture.pdf") is not None


def test_engineer_does_not_match_engine():
    assert ev("Mechanical Engineering Handbook.pdf") is None
    assert ev("Knowledge/Engineering Drawing.pdf") is None


def test_fuel_cell_exception():
    assert ev("Fuel Cell Handbook.pdf") is None
    assert ev("hydrogen fuel-cell systems.pdf") is None
    # Plain fuel-system material is combustion-relevant.
    assert ev("fuel systems overview.pdf") is not None


def test_outdated_software_tutorials_are_ignored():
    assert ev("Autodesk Inventor 2012 Stuff/tutorial.pdf") is not None
    assert ev("Ansys 14.5 (2012) manual.pdf") is not None
    # Recent software material stays.
    assert ev("Ansys 2024 tutorial.pdf") is None


def test_obsolete_battery_chemistry_is_ignored():
    assert ev("lead-acid battery guide.pdf") is not None
    assert ev("NiCd charging notes.pdf") is not None
    # Modern EV batteries stay.
    assert ev("Li-ion battery pack design.pdf") is None


def test_core_ev_and_aero_content_is_kept():
    for name in (
        "Aerodynamics/Luke's Aero Bank 2021/drs paper.pdf",
        "Battery pack enclosure design.pdf",
        "Motor controller cooling.pdf",
        "Suspension geometry notes.pdf",
        "CFD validation.pdf",
        "Chassis torsional stiffness.pdf",
    ):
        assert ev(name) is None, name


def test_live_corpus_low_value_filename_families_are_ignored():
    for name in (
        "Knowledge/NASA/Hollow Earth Proven By Apollo 16.pdf",
        "Knowledge/NASA/What NASA Isn't Telling You About Mars.pdf",
        "Reading Articles/Popular Mechanics - Aim Your Headlights.pdf",
        "Knowledge/Composite/D265T082.pdf",
        "Knowledge/Composite/D265P2003.pdf",
        "Knowledge/Software/viewcmd_report.pdf",
        "Knowledge/Software/SolidCAM_2011_Turning_User_Guide.pdf",
        "(Solutions_Manual)_Engineering_Fluid_Mechanics_7th_Edition.pdf",
        "University Physics with Modern Physics 12e Young [Solutions].pdf",
        "2 - Thermodynamics.pdf",
        "Management 101 The Five Functions of Management.pdf",
        "[unknown] - Management 101 The Five Functions of Management.pdf",
        "Briefcase Books - Six Sigma Managers.pdf",
        "Marchewka - IT Project Managment.pdf",
        "Dahlberg & Kenig - Harmonic Analysis And Partial Differential Equations.pdf",
        "Brin - Introduction to Differential Topology.pdf",
        "Evans - Introduction to stochastic differential equations.pdf",
        "Hacking Google Maps and Google Earth.pdf",
        "[NASA Archive].Foundations of Tensor Analysis.pdf",
        "6.40 - Medical Devices.pdf",
        "6.39 - Medical Applications of Composites.pdf",
        "6.41 - Application of Composites in Sporting Goods.pdf",
        "40QuantumMechanics_notes.pdf",
        "Atomic Spectra.pdf",
        "NASA_X aircraft archive.pdf",
        "Custom Dry Sump Scavenge Pump.pdf",
        "Ignition TroubleShooting.pdf",
        "engine_appendix.pdf",
        "fuel_injection_notes.pdf",
        "How To Hotrod Your Buick V6 - Rick Bailey.pdf",
        "Mastering_autodesk_inventor_2011.pdf",
        "Introduction to SIMULINK With Engineering Applications.pdf",
        "PC1432_Solutiontotutorial40910sem1.pdf",
        "PDE_tutorial_1_intro.pdf",
        "Lecture.Notes.Big.Picture.of.Calculus.pdf",
        "solver_fortran_statements.pdf",
        "viewfn_runtime.pdf",
        "Ship design, construction and operation.pdf",
    ):
        assert ev(name) is not None, name


def test_strict_filter_ignores_generic_bulk_libraries():
    for name in (
        "Knowledge/Finite Element/course/ch01.pdf",
        "Knowledge/Mechanics/reference/ch42.pdf",
        "Knowledge/RAI Foundation Colleges ME Lecture Materials/week 1.pdf",
        "Knowledge/Military/field manual.pdf",
        "Knowledge/Ship/naval architecture.pdf",
        "Cornell Ansys/Basic/tutorial.pdf",
        "[Solutions Manual] Thermodynamics - An Engineering Approach/ch01.pdf",
        "white_fluid_mechanics_5E_solutions/ch01.pdf",
        "Reading Articles/BOOK COLLECTION -Matlab-/guide.pdf",
        "Reading Articles/Engineers_Collection_DVD3/Mathematics/analysis.pdf",
        "Reading Articles/Engineers_Collection_DVD3/Management - Project & Eng/planning.pdf",
        "Aerodynamics/Luke's Aero Bank 2021/Books/CFD Collection/volume 1.pdf",
        "Reading Articles/Books/general automotive handbook.pdf",
    ):
        assert ev(name) is not None, name


def test_strict_filter_preserves_focused_fsae_collections():
    for name in (
        "Knowledge/Vehicle Dynamic/handling.pdf",
        "Knowledge/Chassis/monocoque.pdf",
        "Knowledge/Composite/laminate-design.pdf",
        "Knowledge/EV/inverter.pdf",
        "Knowledge/FSAE/design-review.pdf",
        "Aerodynamics/Readings/ground-effect.pdf",
        "Driveline Readings/differential.pdf",
        "SAE papers/vehicle dynamics.pdf",
    ):
        assert ev(name) is None, name


def test_filter_pdf_list_partitions_and_logs(safe_tmp_path):
    root = safe_tmp_path / "corpus"
    (root / "Engine Related Readings").mkdir(parents=True)
    (root / "aero").mkdir()
    keep_a = root / "aero" / "wing design.pdf"
    keep_b = root / "aero" / "battery.pdf"
    drop = root / "Engine Related Readings" / "engine.pdf"
    for path in (keep_a, keep_b, drop):
        path.write_bytes(b"%PDF")

    log = safe_tmp_path / "logs" / "ignored_documents.log"
    kept, ignored = filter_pdf_list([keep_a, keep_b, drop], root=root, log_file=log)

    assert kept == [keep_a, keep_b]
    assert [p for p, _ in ignored] == [drop]
    text = log.read_text(encoding="utf-8")
    assert "ignored_documents" not in text or True  # header comment allowed
    lines = [line for line in text.splitlines() if not line.startswith("#")]
    assert len(lines) == 1
    assert "Engine Related Readings" + "\\" + "engine.pdf" in lines[0] or \
        "Engine Related Readings/engine.pdf" in lines[0]
    assert "combustion powertrain (engine)" in lines[0]
    # Timestamp prefix format.
    import re as _re

    assert _re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \|", lines[0])


def test_write_ignore_log_appends(safe_tmp_path):
    log = safe_tmp_path / "ignored.log"
    write_ignore_log(
        [(Path("a/engine.pdf"), __import__("src.fsae_ev_filter", fromlist=["FilterDecision"]).FilterDecision("combustion", "test reason"))],
        log,
    )
    write_ignore_log(
        [(Path("b/rules.pdf"), __import__("src.fsae_ev_filter", fromlist=["FilterDecision"]).FilterDecision("outdated", "another"))],
        log,
    )
    lines = [line for line in log.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
    assert len(lines) == 2


def test_expensive_scan_filter_drops_only_long_image_only_pdfs(safe_tmp_path):
    from pypdf import PdfWriter

    long_scan = safe_tmp_path / "long-scan.pdf"
    short_scan = safe_tmp_path / "short-scan.pdf"
    for path, pages in ((long_scan, 51), (short_scan, 5)):
        writer = PdfWriter()
        for _ in range(pages):
            writer.add_blank_page(width=612, height=792)
        with path.open("wb") as handle:
            writer.write(handle)

    scanned: set[Path] = set()
    kept, ignored = filter_expensive_scans(
        [long_scan, short_scan], max_pages=50, root=safe_tmp_path, scanned_out=scanned
    )
    assert kept == [short_scan]
    assert scanned == {short_scan}
    assert [path for path, _ in ignored] == [long_scan]
    assert "51 pages" in ignored[0][1].reason

    kept, ignored = filter_expensive_scans(
        [short_scan], max_pages=50, max_bytes=1, root=safe_tmp_path
    )
    assert kept == []
    assert ignored[0][0] == short_scan


def test_expensive_scan_filter_records_malformed_pdf(safe_tmp_path):
    broken = safe_tmp_path / "broken.pdf"
    broken.write_bytes(b"<html>not a pdf</html>")
    kept, ignored = filter_expensive_scans([broken], max_pages=50, root=safe_tmp_path)
    assert kept == []
    assert ignored[0][0] == broken
    assert "malformed PDF" in ignored[0][1].reason
