const KEY = "onboarding_done";
export function hasCompletedOnboarding() {
    return localStorage.getItem(KEY) === "true";
}
export function setCompletedOnboarding() {
    localStorage.setItem(KEY, "true");
}
export function resetOnboarding() {
    localStorage.removeItem(KEY);
}
