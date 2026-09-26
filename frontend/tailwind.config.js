/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: 'rgb(var(--bg) / <alpha-value>)',
        panel: 'rgb(var(--panel) / <alpha-value>)',
        panel2: 'rgb(var(--panel2) / <alpha-value>)',
        panel3: 'rgb(var(--panel3) / <alpha-value>)',
        borderline: 'rgb(var(--borderline) / <alpha-value>)',
        soft: 'var(--soft)',
        healthy: '#22c55e',
        warning: '#eab308',
        critical: '#ef4444',
        info: '#3b82f6',
      },
      textColor: {
        slate: {
          50: 'rgb(var(--ts-50) / <alpha-value>)',
          100: 'rgb(var(--ts-100) / <alpha-value>)',
          200: 'rgb(var(--ts-200) / <alpha-value>)',
          300: 'rgb(var(--ts-300) / <alpha-value>)',
          400: 'rgb(var(--ts-400) / <alpha-value>)',
          500: 'rgb(var(--ts-500) / <alpha-value>)',
          600: 'rgb(var(--ts-600) / <alpha-value>)',
        },
        sky: {
          200: 'rgb(var(--tsk-200) / <alpha-value>)',
          300: 'rgb(var(--tsk-300) / <alpha-value>)',
          400: 'rgb(var(--tsk-400) / <alpha-value>)',
          500: 'rgb(var(--tsk-500) / <alpha-value>)',
        },
        red: {
          300: 'rgb(var(--tr-300) / <alpha-value>)',
          400: 'rgb(var(--tr-400) / <alpha-value>)',
        },
        amber: {
          300: 'rgb(var(--tam-300) / <alpha-value>)',
          400: 'rgb(var(--tam-400) / <alpha-value>)',
        },
        yellow: {
          300: 'rgb(var(--ty-300) / <alpha-value>)',
          400: 'rgb(var(--ty-400) / <alpha-value>)',
        },
        orange: {
          300: 'rgb(var(--tor-300) / <alpha-value>)',
          400: 'rgb(var(--tor-400) / <alpha-value>)',
        },
        green: {
          300: 'rgb(var(--tg-300) / <alpha-value>)',
          400: 'rgb(var(--tg-400) / <alpha-value>)',
        },
        emerald: {
          300: 'rgb(var(--tem-300) / <alpha-value>)',
          400: 'rgb(var(--tem-400) / <alpha-value>)',
        },
        violet: {
          300: 'rgb(var(--tv-300) / <alpha-value>)',
          400: 'rgb(var(--tv-400) / <alpha-value>)',
        },
        indigo: {
          300: 'rgb(var(--tin-300) / <alpha-value>)',
          400: 'rgb(var(--tin-400) / <alpha-value>)',
        },
        blue: {
          300: 'rgb(var(--tb-300) / <alpha-value>)',
          400: 'rgb(var(--tb-400) / <alpha-value>)',
        },
      },
      fontFamily: {
        sans: [
          'Inter',
          'ui-sans-serif',
          'system-ui',
          '-apple-system',
          'Segoe UI',
          'sans-serif',
        ],
        mono: [
          '"JetBrains Mono"',
          'ui-monospace',
          'SFMono-Regular',
          'Menlo',
          'Consolas',
          'monospace',
        ],
      },
      boxShadow: {
        card: 'inset 0 1px 0 0 rgba(255,255,255,0.04), 0 10px 30px -14px rgba(0,0,0,0.55)',
        cardHover: 'inset 0 1px 0 0 rgba(255,255,255,0.05), 0 18px 40px -18px rgba(0,0,0,0.7)',
        glow: '0 0 0 1px rgba(56,189,248,0.18), 0 12px 40px -12px rgba(56,189,248,0.25)',
      },
      backgroundImage: {
        hero: 'radial-gradient(55rem 26rem at 12% -12%, var(--hero-1), transparent 60%), radial-gradient(45rem 24rem at 92% -14%, var(--hero-2), transparent 60%)',
      },
      keyframes: {
        'fade-up': {
          from: { opacity: '0', transform: 'translateY(8px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        'pulse-soft': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '.45' },
        },
        shimmer: {
          '0%': { backgroundPosition: '200% 0' },
          '100%': { backgroundPosition: '-200% 0' },
        },
      },
      animation: {
        'fade-up': 'fade-up .4s ease-out both',
        'pulse-soft': 'pulse-soft 2s ease-in-out infinite',
      },
    },
  },
  plugins: [],
};