import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import sitemap from '@astrojs/sitemap';

// docs.glacis.io — company docs aligned to labs product + glacis-web-prod brand.
//
// IA choice (2026-09-27): keep the Start → Connect activation ladder from
// origin/main (honest SDK + portal onboarding), and add company-wide sections
// that labs + brand require: Runtime, OVERT (orientation only; normative text
// stays at overt.is), OVERT-as-Code (Preview), and Concepts.
// Rejected pure /sdk-only and pure /start+/connect-only trees — both leave the
// runtime product and standard under-documented relative to glacis.io.
//
// Vocabulary: marketing-facing copy prefers "record"; "receipt" reserved for
// OVERT/spec field names and SDK wire types (matches glacis-web-prod).

export default defineConfig({
  site: 'https://docs.glacis.io',
  trailingSlash: 'always',
  redirects: {
    // Legacy SDK paths (PyPI / older docs) → Connect
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
        light: './src/assets/glacis-wordmark.svg',
        dark: './src/assets/glacis-wordmark-dark.svg',
        replacesTitle: true,
        alt: 'GLACIS',
      },
      favicon: '/favicon.ico',
      head: [
        { tag: 'link', attrs: { rel: 'apple-touch-icon', href: '/favicons/apple-touch-icon.png' } },
        { tag: 'link', attrs: { rel: 'icon', type: 'image/png', sizes: '32x32', href: '/favicons/favicon-32.png' } },
        { tag: 'meta', attrs: { name: 'theme-color', content: '#18141F' } },
        { tag: 'meta', attrs: { property: 'og:image', content: 'https://docs.glacis.io/og-default.png' } },
        { tag: 'meta', attrs: { name: 'twitter:card', content: 'summary_large_image' } },
        { tag: 'meta', attrs: { name: 'twitter:image', content: 'https://docs.glacis.io/og-default.png' } },
      ],
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
          label: 'Runtime product',
          items: [
            { label: 'Overview', link: '/runtime/' },
            { label: 'Inspect under NDA', link: '/runtime/inspect-under-nda/' },
          ],
        },
        {
          label: 'OVERT — the standard',
          items: [
            { label: 'Overview', link: '/overt/' },
            { label: 'Conformance ladder', link: '/overt/conformance-ladder/' },
          ],
        },
        {
          label: 'OVERT-as-Code',
          badge: { text: 'Preview', variant: 'caution' },
          items: [
            { label: 'Overview', link: '/overt-as-code/' },
            { label: 'Quickstart', link: '/overt-as-code/quickstart/' },
            { label: 'Policy as code for AI', link: '/overt-as-code/policy-as-code-for-ai/' },
            { label: 'OSCAL export', link: '/overt-as-code/oscal-export/' },
          ],
        },
        {
          label: 'Concepts',
          items: [
            { label: 'Documentation is not evidence', link: '/concepts/documentation-is-not-evidence/' },
            { label: 'AI attestation, explained', link: '/concepts/ai-attestation-explained/' },
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
    sitemap(),
  ],
});
