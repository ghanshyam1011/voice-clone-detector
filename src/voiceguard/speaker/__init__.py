"""Speaker-consistency branch.

`SpeakerEmbedder` turns a waveform into a speaker embedding (pretrained,
inference only). `SpeakerVerifier` holds per-session enrolment embeddings
and scores a live window against them: cosine similarity -> a mismatch
risk in [0, 1].

This answers "is this the *same person* who enrolled", which is separate
from "is this synthetic" — an attacker can clone the right voice, or use
a real voice that isn't the customer.
"""

from voiceguard.speaker.embed import SpeakerEmbedder
from voiceguard.speaker.verify import SpeakerVerifier

__all__ = ["SpeakerEmbedder", "SpeakerVerifier"]
