// tailwind.config.js
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: '#F9FAFB', // Off-white lembut
        surface: '#FFFFFF',
        primary: '#5B7DB1',    // Muted Blue-Gray
        secondary: '#9CA3AF',
        accent: {
          calm: '#BFDBFE',     // Biru lembut
          sad: '#E9D5FF',      // Lavender
          anxious: '#FDE68A',  // Krem hangat
        }
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'], // Pastikan import font Inter di CSS
        heading: ['Poppins', 'sans-serif'],
      },
      animation: {
        'breathe': 'breathe 3s ease-in-out infinite',
      },
      keyframes: {
        breathe: {
          '0%, 100%': { transform: 'scale(1)' },
          '50%': { transform: 'scale(1.02)' },
        }
      }
    },
  },
  plugins: [],
}