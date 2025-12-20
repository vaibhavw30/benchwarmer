/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // NBA-themed dark mode colors
        'nba-dark': '#0F172A',
        'nba-darker': '#0A0F1E',
        'nba-blue': '#1E40AF',
        'nba-red': '#DC2626',
        'nba-orange': '#EA580C',
        'nba-purple': '#7C3AED',
      },
    },
  },
  plugins: [],
}
