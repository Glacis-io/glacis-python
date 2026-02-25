# GLACIS Documentation Portal

This directory contains the source for the GLACIS documentation site at [docs.glacis.io](https://docs.glacis.io).

## Technology Stack

- **Framework**: [Astro](https://astro.build/) with [Starlight](https://starlight.astro.build/)
- **Hosting**: Cloudflare Pages
- **Content**: MDX (Markdown with components)

## Development

### Prerequisites

- Node.js 18+
- pnpm (recommended) or npm

### Getting Started

```bash
# Install dependencies
pnpm install

# Start development server
pnpm dev

# Build for production
pnpm build

# Preview production build
pnpm preview
```

### Development Server

The dev server runs at `http://localhost:4321` with hot reload.

## Structure

```
docs/
├── astro.config.mjs     # Astro + Starlight configuration
├── package.json         # Dependencies
├── tsconfig.json        # TypeScript configuration
├── public/              # Static assets
│   ├── robots.txt       # SEO configuration
│   ├── og-image.svg
│   └── glacis-convert.js
└── src/
    ├── assets/          # Images and SVGs
    │   ├── glacis-logo.svg
    │   ├── glacis-logo.png
    │   └── glacis-hero.svg
    ├── content/
    │   ├── config.ts    # Content schema
    │   └── docs/        # Documentation pages (MDX)
    │       ├── index.mdx
    │       └── sdk/
    │           └── python/
    │               ├── installation.mdx
    │               ├── quickstart.mdx
    │               ├── openai.mdx
    │               ├── anthropic.mdx
    │               ├── gemini.mdx
    │               ├── offline.mdx
    │               ├── controls.mdx
    │               ├── configuration.mdx
    │               ├── sampling.mdx
    │               ├── judges.mdx
    │               ├── batch.mdx
    │               ├── storage.mdx
    │               ├── cli.mdx
    │               └── api.mdx
    └── styles/
        └── custom.css   # Custom styling
```

## Content Guidelines

### Adding New Pages

1. Create a new `.mdx` file in the appropriate directory
2. Add frontmatter with `title` and `description`
3. Add the page to the sidebar in `astro.config.mjs`

### Frontmatter

```mdx
---
title: Page Title
description: SEO description for the page.
---
```

### Using Components

Starlight provides built-in components:

```mdx
import { Card, CardGrid, LinkCard, Aside, Tabs, TabItem, Steps } from '@astrojs/starlight/components';

<Card title="Feature" icon="rocket">
  Card content here.
</Card>

<Aside type="tip">
  Helpful tip here.
</Aside>

<Steps>
1. First step
2. Second step
</Steps>
```

### Code Blocks

Use fenced code blocks with language hints:

```typescript
const example = 'code';
```

## Deployment

The docs site is automatically deployed to Cloudflare Pages when changes are pushed to the `main` branch.

### Manual Deployment

```bash
# Build
pnpm build

# Deploy to Cloudflare Pages
npx wrangler pages deploy dist
```

## SEO

- All pages should have descriptive `title` and `description` frontmatter
- The sitemap is automatically generated at `/sitemap-index.xml`
- Schema.org structured data is included for software documentation
- Open Graph and Twitter cards are configured

## Contributing

1. Create a branch for your changes
2. Make edits to the MDX files
3. Test locally with `pnpm dev`
4. Submit a pull request

## Links

- **Live Site**: https://docs.glacis.io
- **Main Repository**: https://github.com/Glacis-io/glacis-python
- **Discord**: https://discord.gg/glacis
