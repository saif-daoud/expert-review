import pytest

from server.session_rules import farewell_phrase


@pytest.mark.parametrize(
    "text",
    (
        "Patient: Okay. Um... bye.",
        "Therapist: Bye-bye.",
        "Take care. Bye bye.",
        "good bye",
        "good-bye",
        "goodbye",
    ),
)
def test_simulation_farewells_are_recognized(text):
    assert farewell_phrase(text) is not None


@pytest.mark.parametrize(
    "text",
    (
        "This should bypass the issue.",
        "That was a good byproduct.",
        "I said goodbye to my father.",
        "It can be hard to say goodbye.",
        "The word goodbye feels final.",
    ),
)
def test_simulation_farewell_false_positives_are_ignored(text):
    assert farewell_phrase(text) is None
