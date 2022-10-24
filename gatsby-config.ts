export const siteMetadata = {
  title: "Dylan's Blog",
  author: 'Dylan Djian',
  description: 'A blog on my personal interests and findings',
  siteUrl: 'https://dylandjian.github.io',
}

export const plugins = [
  {
    resolve: `gatsby-omni-font-loader`,
    options: {
      enableListener: true,
      preconnect: [`https://fonts.googleapis.com`, `https://fonts.gstatic.com`],
      web: [
        {
          name: `Merriweather`,
          file: `https://fonts.googleapis.com/css2?family=Merriweather:wght@400;600;700&display=swap`,
        },
      ],
    },
  },
  {
    resolve: '@chakra-ui/gatsby-plugin',
    options: {
      resetCSS: false,
      isUsingColorMode: true,
    },
  },
  'gatsby-plugin-image',
  {
    resolve: 'gatsby-source-filesystem',
    options: {
      path: `${__dirname}/content/blog`,
      name: 'blog',
    },
  },
  {
    resolve: 'gatsby-source-filesystem',
    options: {
      name: 'images',
      path: `${__dirname}/src/images`,
    },
  },
  {
    resolve: 'gatsby-transformer-remark',
    options: {
      plugins: [
        {
          resolve: 'gatsby-remark-embed-youtube',
          options: {
            width: 800,
            height: 400,
          },
        },
        {
          resolve: 'gatsby-remark-responsive-iframe',
          options: {
            wrapperStyle: 'margin-bottom: 1.0725rem',
          },
        },
        'gatsby-remark-copy-linked-files',
        'gatsby-remark-smartypants',
        {
          resolve: 'gatsby-remark-images',
          options: {
            maxWidth: 630,
          },
        },
        {
          resolve: `gatsby-remark-katex`,
          options: {
            // Add any KaTeX options from https://github.com/KaTeX/KaTeX/blob/master/docs/options.md here
            strict: `ignore`,
          },
        },
        'gatsby-remark-prismjs',
      ],
    },
  },
  'gatsby-transformer-sharp',
  'gatsby-plugin-sharp',
  {
    resolve: 'gatsby-plugin-manifest',
    options: {
      name: "Dylan's personal blog",
      short_name: "Dylan's blog",
      start_url: '/',
      background_color: '#ffffff',
      // This will impact how browsers show your PWA/website
      // https://css-tricks.com/meta-theme-color-and-trickery/
      // theme_color: `#663399`,
      display: 'minimal-ui',
      icon: 'src/images/gatsby-icon.png', // This path is relative to the root of the site.
    },
  },
]
