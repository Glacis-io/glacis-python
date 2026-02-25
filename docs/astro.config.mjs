import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  site: 'https://docs.glacis.io',
  trailingSlash: 'always',
  integrations: [
    starlight({
      title: 'GLACIS',
      logo: {
        src: './src/assets/glacis-logo.png',
        alt: 'GLACIS Logo',
      },
      social: {
        github: 'https://github.com/Glacis-io/glacis-python',
      },
      editLink: {
        baseUrl: 'https://github.com/Glacis-io/glacis-python/edit/main/docs/',
      },
      customCss: ['./src/styles/custom.css'],
      sidebar: [
        {
          label: 'Getting Started',
          items: [
            { label: 'Installation', link: '/sdk/python/installation/' },
            { label: 'Quickstart', link: '/sdk/python/quickstart/' },
            { label: 'Configuration', link: '/sdk/python/configuration/' },
          ],
        },
        {
          label: 'Integrations',
          items: [
            { label: 'OpenAI', link: '/sdk/python/openai/' },
            { label: 'Anthropic', link: '/sdk/python/anthropic/' },
            { label: 'Gemini', link: '/sdk/python/gemini/' },
          ],
        },
        {
          label: 'Features',
          items: [
            { label: 'Offline Mode', link: '/sdk/python/offline/' },
            { label: 'Controls', link: '/sdk/python/controls/' },
            { label: 'Sampling & Evidence', link: '/sdk/python/sampling/' },
            { label: 'Judges Framework', link: '/sdk/python/judges/' },
            { label: 'Batch Operations', link: '/sdk/python/batch/' },
            { label: 'Storage Backends', link: '/sdk/python/storage/' },
          ],
        },
        {
          label: 'Reference',
          items: [
            { label: 'CLI', link: '/sdk/python/cli/' },
            { label: 'API Reference', link: '/sdk/python/api/' },
          ],
        },
      ],
    }),
  ],
});
