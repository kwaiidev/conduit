import React from "react";
import { enableASL, getASLReady } from "../../lib/aslcv";
import { setModalityPreference } from "../../state/modalityPreferences";

const ASL_VIDEO_URL = "http://localhost:8765/video";

type HealthState = {
  signBackend: boolean;
  aslReady: boolean;
  cvBackend: boolean;
};

function describeHealth(health: HealthState): string {
  if (!health.signBackend) {
    return "Sign backend is offline. Start ASL service to test visuals.";
  }
  if (!health.aslReady) {
    return "Sign backend is running, but ASL detection is not active yet.";
  }
  if (!health.cvBackend) {
    return "ASL is active. Start eye-gaze CV backend to validate both together.";
  }
  return "ASL and eye-gaze CV are both running. You can validate in Visualizations.";
}

async function sleep(ms: number): Promise<void> {
  await new Promise(resolve => window.setTimeout(resolve, ms));
}

export function ASLVisualsCheckStep() {
  const [health, setHealth] = React.useState<HealthState>({
    signBackend: false,
    aslReady: false,
    cvBackend: false,
  });
  const [statusMessage, setStatusMessage] = React.useState(
    "Start ASL service, then verify ASL + CV health together."
  );
  const [isStartingAsl, setIsStartingAsl] = React.useState(false);
  const [isStartingCv, setIsStartingCv] = React.useState(false);
  const [videoError, setVideoError] = React.useState(false);
  const [videoNonce, setVideoNonce] = React.useState(() => Date.now());

  const probeStatus = React.useCallback(async (silent = false): Promise<HealthState> => {
    const [signBackend, cvBackend] = await Promise.all([
      window.electron.isSignBackendRunning(),
      window.electron.isCvBackendRunning(),
    ]);

    let aslReady = false;
    if (signBackend) {
      aslReady = await getASLReady();
    }

    const nextHealth = { signBackend, aslReady, cvBackend };
    setHealth(nextHealth);
    if (!silent) {
      setStatusMessage(describeHealth(nextHealth));
    }
    return nextHealth;
  }, []);

  const startAslService = React.useCallback(async () => {
    setIsStartingAsl(true);
    setStatusMessage("Launching sign backend...");
    try {
      const launchResult = await window.electron.startSignBackend();
      if (!launchResult.ok) {
        setStatusMessage(`Unable to launch sign backend: ${launchResult.message}`);
        setModalityPreference("sign-text", false);
        return;
      }

      setStatusMessage("Enabling ASL detection and session...");
      await enableASL();

      const deadline = Date.now() + 12000;
      let ready = false;
      while (Date.now() < deadline) {
        ready = await getASLReady();
        if (ready) {
          break;
        }
        await sleep(400);
      }
      setModalityPreference("sign-text", ready);

      const nextHealth = await probeStatus(true);
      if (ready && nextHealth.cvBackend) {
        setStatusMessage("ASL and eye-gaze CV are both active. Open Visualizations to confirm live feed.");
      } else if (ready) {
        setStatusMessage("ASL is active. Start eye-gaze CV backend to validate both together.");
      } else {
        setStatusMessage("ASL backend started, but detector is not ready yet. Wait a moment and refresh.");
      }
      setVideoError(false);
      setVideoNonce(Date.now());
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      setStatusMessage(`ASL startup failed: ${detail}`);
      setModalityPreference("sign-text", false);
    } finally {
      setIsStartingAsl(false);
    }
  }, [probeStatus]);

  const startCvService = React.useCallback(async () => {
    setIsStartingCv(true);
    setStatusMessage("Launching eye-gaze CV backend...");
    try {
      const launchResult = await window.electron.startCvBackend({ camera: 0 });
      if (!launchResult.ok) {
        setStatusMessage(`Unable to launch eye-gaze CV backend: ${launchResult.message}`);
        setModalityPreference("cv-pointer", false);
        return;
      }

      const nextHealth = await probeStatus(true);
      setModalityPreference("cv-pointer", nextHealth.cvBackend);
      if (nextHealth.cvBackend && nextHealth.aslReady) {
        setStatusMessage("ASL and eye-gaze CV are both active. You can verify both in the UI now.");
      } else if (nextHealth.cvBackend) {
        setStatusMessage("Eye-gaze CV is active. Start ASL service to complete multimodal validation.");
      } else {
        setStatusMessage("Eye-gaze CV launch requested. Refresh status if it is still warming up.");
      }
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      setStatusMessage(`CV startup failed: ${detail}`);
      setModalityPreference("cv-pointer", false);
    } finally {
      setIsStartingCv(false);
    }
  }, [probeStatus]);

  React.useEffect(() => {
    void probeStatus();
    const timer = window.setInterval(() => {
      void probeStatus();
    }, 3000);
    return () => window.clearInterval(timer);
  }, [probeStatus]);

  React.useEffect(() => {
    if (!health.signBackend) {
      setVideoError(false);
      return;
    }
    setVideoError(false);
    setVideoNonce(Date.now());
  }, [health.signBackend]);

  return (
    <div style={styles.container}>
      <div style={styles.card}>
        <h3 style={styles.title}>ASL + Eye-Gaze Validation</h3>
        <p style={styles.description}>
          This step starts the sign backend, enables ASL inference, and checks whether eye-gaze CV is online too.
        </p>

        <div style={styles.badgeRow}>
          <StatusBadge label="Sign Backend" online={health.signBackend} />
          <StatusBadge label="ASL Detection" online={health.aslReady} />
          <StatusBadge label="CV Backend" online={health.cvBackend} />
        </div>

        <div style={styles.buttonRow}>
          <button
            type="button"
            onClick={startAslService}
            style={{
              ...styles.button,
              ...(isStartingAsl ? styles.buttonDisabled : {}),
            }}
            disabled={isStartingAsl}
          >
            {isStartingAsl ? "Starting ASL..." : "Start ASL Service"}
          </button>
          <button
            type="button"
            onClick={startCvService}
            style={{
              ...styles.buttonSecondary,
              ...(isStartingCv ? styles.buttonDisabled : {}),
            }}
            disabled={isStartingCv}
          >
            {isStartingCv ? "Starting CV..." : "Start CV Service"}
          </button>
          <button type="button" onClick={() => void probeStatus()} style={styles.buttonTertiary}>
            Refresh Status
          </button>
        </div>

        <div style={styles.previewOuter}>
          {health.signBackend && !videoError ? (
            <img
              key={videoNonce}
              src={ASL_VIDEO_URL}
              alt="ASL live feed"
              style={styles.previewImage}
              onError={() => setVideoError(true)}
            />
          ) : (
            <div style={styles.previewPlaceholder}>
              {health.signBackend
                ? "ASL video feed unavailable. Check ASL backend logs."
                : "ASL video feed appears after sign backend starts."}
            </div>
          )}
        </div>

        <p style={styles.status}>{statusMessage}</p>
      </div>
    </div>
  );
}

function StatusBadge({ label, online }: { label: string; online: boolean }) {
  return (
    <div
      style={{
        ...styles.badge,
        color: online ? "#16a34a" : "#9ca3af",
        borderColor: online ? "rgba(22,163,74,0.28)" : "rgba(156,163,175,0.22)",
        background: online ? "rgba(22,163,74,0.10)" : "rgba(156,163,175,0.10)",
      }}
    >
      {online ? "●" : "○"} {label}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: "flex",
    justifyContent: "center",
    width: "100%",
  },
  card: {
    width: "100%",
    maxWidth: 520,
    borderRadius: 18,
    border: "1px solid var(--border)",
    background: "var(--bg-secondary)",
    padding: 18,
    display: "flex",
    flexDirection: "column",
    gap: 10,
  },
  title: {
    margin: 0,
    fontSize: 18,
    color: "var(--text-primary)",
  },
  description: {
    margin: 0,
    fontSize: 13,
    color: "var(--text-secondary)",
  },
  badgeRow: {
    display: "flex",
    gap: 10,
    flexWrap: "wrap",
  },
  badge: {
    fontSize: 12,
    fontWeight: 900,
    borderRadius: 999,
    border: "1px solid transparent",
    padding: "7px 10px",
    whiteSpace: "nowrap",
  },
  buttonRow: {
    display: "flex",
    gap: 10,
    flexWrap: "wrap",
  },
  button: {
    height: 38,
    padding: "0 14px",
    borderRadius: 12,
    border: "1px solid rgba(255,45,141,0.35)",
    background: "#FF2D8D",
    color: "#fff",
    fontWeight: 800,
    cursor: "pointer",
  },
  buttonSecondary: {
    height: 38,
    padding: "0 14px",
    borderRadius: 12,
    border: "1px solid var(--border)",
    background: "var(--bg-tertiary)",
    color: "var(--text-primary)",
    fontWeight: 800,
    cursor: "pointer",
  },
  buttonTertiary: {
    height: 38,
    padding: "0 14px",
    borderRadius: 12,
    border: "1px solid var(--border)",
    background: "transparent",
    color: "var(--text-secondary)",
    fontWeight: 700,
    cursor: "pointer",
  },
  buttonDisabled: {
    opacity: 0.65,
    cursor: "default",
  },
  previewOuter: {
    marginTop: 2,
    borderRadius: 16,
    border: "1px solid var(--border)",
    background: "var(--bg-tertiary)",
    overflow: "hidden",
    minHeight: 184,
    display: "grid",
    placeItems: "center",
  },
  previewImage: {
    width: "100%",
    height: "100%",
    minHeight: 184,
    objectFit: "cover",
    display: "block",
  },
  previewPlaceholder: {
    fontSize: 12,
    fontWeight: 700,
    color: "var(--text-secondary)",
    letterSpacing: "0.02em",
    textAlign: "center",
    padding: "16px 20px",
  },
  status: {
    margin: 0,
    minHeight: 18,
    fontSize: 12,
    color: "var(--text-secondary)",
  },
};
