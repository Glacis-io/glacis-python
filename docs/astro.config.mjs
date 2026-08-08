import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// The information architecture follows the activation ladder:
//   Start   — no-code, phone-legible, browser only
//   Connect — the SDK, once receipts should come from your own system
//   Verify  — what anyone can check, and what a check does not establish
//   OVERT   — the open standard underneath the format
//   Reference — the full SDK surface
//
// The old /sdk/python/* URLs are redirected below rather than dropped: they are
// linked from PyPI and from the package metadata.
export default defineConfig({
  site: 'https://docs.glacis.io',
  trailingSlash: 'always',
  redirects: {
    '/sdk/python/': '/connect/',
    '/sdk/python/installation/': '/connect/install/',
    '/sdk/python/quickstart/': '/connect/quickstart/',
    '/sdk/python/configuration/': '/connect/configuration/',
    '/sdk/python/offline/': '/connect/offline-vs-witnessed/',
    '/sdk/python/openai/': '/connect/openai/',
    '/sdk/python/anthropic/': '/connect/anthropic/',
    '/sdk/python/gemini/': '/connect/gemini/',
    '/sdk/python/litellm/': '/connect/litellm/',
    '/sdk/python/cli/': '/verify/cli/',
    '/sdk/python/api/': '/reference/api/',
    '/sdk/python/controls/': '/reference/controls/',
    '/sdk/python/sampling/': '/reference/sampling-and-evidence/',
    '/sdk/python/storage/': '/reference/storage/',
    '/sdk/python/judges/': '/reference/judges/',
    '/sdk/python/batch/': '/reference/operations/',
    '/sdk/python/pipelines/': '/reference/operations/',
  },
  integrations: [
    starlight({
      title: 'GLACIS',
      logo: {
        src: './src/assets/glacis-logo.png',
        alt: 'GLACIS Logo',
      },
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/Glacis-io/glacis-python' },
      ],
      editLink: {
        baseUrl: 'https://github.com/Glacis-io/glacis-python/edit/main/docs/',
      },
      customCss: ['./src/styles/custom.css'],
      sidebar: [
        {
          label: 'Start — no code',
          autogenerate: { directory: 'start' },
        },
        {
          label: 'Connect — the SDK',
          autogenerate: { directory: 'connect' },
        },
        {
          label: 'Verify',
          autogenerate: { directory: 'verify' },
        },
        {
          label: 'OVERT standard',
          items: [
            { label: 'OVERT', link: '/overt/' },
          ],
        },
        {
          label: 'Reference',
          items: [
            { label: 'API reference', link: '/reference/api/' },
            { label: 'Controls', link: '/reference/controls/' },
            { label: 'Sampling & evidence', link: '/reference/sampling-and-evidence/' },
            { label: 'Storage', link: '/reference/storage/' },
            { label: 'Operations & linking', link: '/reference/operations/' },
            { label: 'Judges', link: '/reference/judges/' },
          ],
        },
      ],
    }),
  ],
});
