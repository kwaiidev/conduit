const MODALITY_STORAGE_KEYS = {
    "cv-pointer": "conduit-modality-intent-cv-pointer",
    "eeg-select": "conduit-modality-intent-eeg-select",
    "voice-text": "conduit-modality-intent-voice-text",
    "sign-text": "conduit-modality-intent-sign-text",
};
const ALL_MODALITY_FEATURE_IDS = [
    "cv-pointer",
    "eeg-select",
    "voice-text",
    "sign-text",
];
function canUseLocalStorage() {
    return typeof window !== "undefined" && Boolean(window.localStorage);
}
export function getModalityPreference(featureId) {
    if (!canUseLocalStorage()) {
        return null;
    }
    const rawValue = window.localStorage.getItem(MODALITY_STORAGE_KEYS[featureId]);
    if (rawValue === null) {
        return null;
    }
    return rawValue === "1";
}
export function setModalityPreference(featureId, enabled) {
    if (!canUseLocalStorage()) {
        return;
    }
    window.localStorage.setItem(MODALITY_STORAGE_KEYS[featureId], enabled ? "1" : "0");
}
export function resetAllModalityPreferences(enabled = false) {
    for (const featureId of ALL_MODALITY_FEATURE_IDS) {
        setModalityPreference(featureId, enabled);
    }
}
export function getPreferredActiveModes() {
    return ALL_MODALITY_FEATURE_IDS.filter((featureId) => getModalityPreference(featureId) === true);
}
