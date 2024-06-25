import path from 'path'
import { createFilePath } from 'gatsby-source-filesystem'

export async function createPages({ graphql, actions, reporter }) {
  const { createPage } = actions

  // Define a template for blog post
  const blogPost = path.resolve('./src/templates/BlogPost.tsx')
  const galleryPost = path.resolve('./src/templates/GalleryPost.tsx')
  const recipe = path.resolve('./src/templates/Recipe.tsx')

  const templateByType = {
    blog: blogPost,
    recipe,
    gallery: galleryPost,
  }

  // Get all markdown blog posts sorted by date
  const result = await graphql(`
    {
      allMarkdownRemark(sort: { frontmatter: { date: ASC } }, limit: 1000) {
        nodes {
          id
          fields {
            slug
          }
        }
      }
    }
  `)

  if (result.errors) {
    reporter.panicOnBuild(
      'There was an error loading your blog posts',
      result.errors,
    )
    return
  }

  const posts = result.data.allMarkdownRemark.nodes

  // Create blog posts pages
  // But only if there's at least one markdown file found at "content/blog" (defined in gatsby-config.js)
  // `context` is available in the template as a prop and as a variable in GraphQL

  if (posts.length > 0) {
    for (const type in templateByType) {
      const relatedPosts = posts.filter((post) => {
        return post.fields && post.fields.slug.includes(type)
      })

      for (const [index, post] of relatedPosts.entries()) {
        const previousPostId = index === 0 ? null : relatedPosts[index - 1].id
        const nextPostId =
          index === relatedPosts.length - 1 ? null : relatedPosts[index + 1].id

        createPage({
          path: post.fields.slug,
          component: templateByType[type],
          context: {
            id: post.id,
            previousPostId,
            nextPostId,
          },
        })
      }
    }
  }
}

function createType(absolutePath: string) {
  if (absolutePath.includes('gallery')) {
    return 'gallery'
  }

  if (absolutePath.includes('blog')) {
    return 'blog'
  }

  return 'recipe'
}

export function onCreateNode({ node, actions, getNode }) {
  const { createNodeField } = actions

  if (node.internal.type === 'MarkdownRemark') {
    const value = createFilePath({ node, getNode })
    const type = createType(node.fileAbsolutePath)

    createNodeField({
      name: 'slug',
      node,
      value: `/${type}${value}`,
    })
    createNodeField({
      name: 'name',
      node,
      value: value.split('/')[1],
    })
    createNodeField({
      name: 'type',
      node,
      value: type,
    })
  }
}

export function createSchemaCustomization({ actions }) {
  const { createTypes } = actions

  // Explicitly define the siteMetadata {} object
  // This way those will always be defined even if removed from gatsby-config.js

  // Also explicitly define the Markdown frontmatter
  // This way the "MarkdownRemark" queries will return `null` even when no
  // blog posts are stored inside "content/blog" instead of returning an error

  createTypes(`
    type SiteSiteMetadata {
      author: Author
      siteUrl: String
      social: Social
    }

    type Author {
      name: String
      summary: String
    }

    type Social {
      twitter: String
    }

    type MarkdownRemark implements Node {
      frontmatter: Frontmatter
      fields: Fields
    }

    type Frontmatter {
      title: String
      description: String
      date: Date @dateformat
    }

    type Fields {
      slug: String
      type: String
      name: String
    }
  `)
}
