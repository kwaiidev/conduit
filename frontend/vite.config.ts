import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  resolve: {
    // Prefer TS sources over emitted JS siblings during local development.
    extensions: [".ts", ".tsx", ".js", ".jsx", ".mjs", ".json"],
  },
  base: "./",
  build: { outDir: "dist" },
});
