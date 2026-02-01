from ttsforge.ssmd_generator import chapter_to_ssmd


def test_emphasis_repeated_phrases() -> None:
    html = "This is <em>very</em> good. This is <em>very</em> good."
    text = "This is very good. This is very good."
    ssmd = chapter_to_ssmd(
        chapter_title="",
        chapter_text=text,
        html_content=html,
        include_title=False,
    )
    assert ssmd.count("*very*") == 2


def test_emphasis_with_punctuation() -> None:
    html = "Wait, <strong>now</strong>."
    text = "Wait, now."
    ssmd = chapter_to_ssmd(
        chapter_title="",
        chapter_text=text,
        html_content=html,
        include_title=False,
    )
    assert "**now**" in ssmd
