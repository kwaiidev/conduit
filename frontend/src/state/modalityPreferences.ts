export type ModalityFeatureId = "cv-pointer" | "eeg-select" | "voice-text" | "sign-text";

const MODALITY_STORAGE_KEYS: Record<ModalityFeatureId, string> = {
  "cv-pointer": "conduit-modality-intent-cv-pointer",
  "eeg-select": "conduit-modality-intent-eeg-select",
  "voice-text": "conduit-modality-intent-voice-text",
  "sign-text": "conduit-modality-intent-sign-text",
};

const ALL_MODALITY_FEATURE_IDS: ModalityFeatureId[] = [
  "cv-pointer",
  "eeg-select",
  "voice-text",
  "sign-text",
];

function canUseLocalStorage(): boolean {
  return typeof window !== "undefined" && Boolean(window.localStorage);
}

export function getModalityPreference(featureId: ModalityFeatureId): boolean | null {
  if (!canUseLocalStorage()) {
    return null;
  }
  const rawValue = window.localStorage.getItem(MODALITY_STORAGE_KEYS[featureId]);
  if (rawValue === null) {
    return null;
  }
  return rawValue === "1";
}

export function setModalityPreference(featureId: ModalityFeatureId, enabled: boolean): void {
  if (!canUseLocalStorage()) {
    return;
  }
  window.localStorage.setItem(MODALITY_STORAGE_KEYS[featureId], enabled ? "1" : "0");
}

export function resetAllModalityPreferences(enabled = false): void {
  for (const featureId of ALL_MODALITY_FEATURE_IDS) {
    setModalityPreference(featureId, enabled);
  }
}

export function getPreferredActiveModes(): ModalityFeatureId[] {
  return ALL_MODALITY_FEATURE_IDS.filter((featureId) => getModalityPreference(featureId) === true);
}
