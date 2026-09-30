"""Drive a VoiceGuard session over gRPC -- the same flow as rest_client.py,
same output, different transport. Compare the two side by side: this is the
proof that gRPC isn't a second implementation, just a second door into the
same SessionManager.

    python scripts/serve_grpc.py                 # in one terminal
    python examples/grpc_client.py enrol.wav benign.wav cloned.wav

Needs the [grpc] extra: pip install -e ".[grpc]"
"""

from __future__ import annotations

import sys

import grpc

from voiceguard.serve.proto import voiceguard_pb2 as pb
from voiceguard.serve.proto import voiceguard_pb2_grpc as pb_grpc

ADDR = "127.0.0.1:50051"


def _read(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles + the rupee sign
    except (AttributeError, ValueError):
        pass
    if len(sys.argv) != 4:
        sys.exit("usage: grpc_client.py <enrol.wav> <benign.wav> <cloned.wav>")
    enrol, benign, cloned = sys.argv[1:4]

    with grpc.insecure_channel(ADDR) as channel:
        stub = pb_grpc.VoiceGuardStub(channel)

        s = stub.CreateSession(pb.CreateSessionRequest(scenario="transaction"))
        sid = s.session_id
        print(f"session {sid}")

        enr = stub.Enroll(pb.EnrollRequest(session_id=sid, audio=_read(enrol), filename=enrol))
        print("enrol:", {"enrolled": enr.enrolled, "seconds": round(enr.seconds, 2)})

        def analyse(path: str, label: str) -> None:
            d = stub.Analyze(pb.AnalyzeRequest(session_id=sid, audio=_read(path), filename=path))
            print(f"\n[{label}]  {d.action}  risk={round(d.fused * 100)}")
            for name in ("spoof", "speaker", "prosody"):
                v = getattr(d, name) if d.HasField(name) else None
                print(f"    {name:8s} {'--' if v is None else round(v * 100)}")
            for reason in d.reasons:
                print(f"    - {reason}")
            print(f"    -> {d.recommendation}")

        analyse(benign, "benign call")

        stub.UpdateContext(
            pb.UpdateContextRequest(session_id=sid, caller_known=False, amount=75000)
        )
        analyse(cloned, "cloned voice + unknown caller + large transfer")

        stub.Acknowledge(pb.AcknowledgeRequest(session_id=sid))
        trail = stub.GetAudit(pb.GetAuditRequest(session_id=sid))
        print(f"\naudit trail ({len(trail.events)} events, feature-only):")
        for e in trail.events:
            print(f"    {e.utc}  {e.action:12s}  fused={e.fused}  {dict(e.signals)}")


if __name__ == "__main__":
    main()
