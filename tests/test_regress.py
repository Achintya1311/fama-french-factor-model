from factors.regress import main


def test_cli_diagnostics_flag_runs_without_error(capsys):
    main(["--portfolio", "hitec", "--model", "ff3", "--diagnostics"])
    captured = capsys.readouterr()
    assert "alpha t-stat (NW)" in captured.out
    # R11: loadings_t_nw was computed by run_diagnostics but never printed
    # here either -- every industry-portfolio loading claim this CLI can
    # produce was OLS-only, with no Newey-West caveat.
    assert "loadings t (NW)" in captured.out
    for factor in ("smb", "hml"):
        assert factor in captured.out.split("loadings t (NW)")[1].split("\n")[0]
