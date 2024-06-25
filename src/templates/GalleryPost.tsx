import * as React from 'react'
import { Link, graphql } from 'gatsby'

import { Bio } from '../components/Bio'
import { Layout } from '../components/Layout'
import { Seo } from '../components/Seo'

import Gallery from '@browniebroke/gatsby-image-gallery'
import { Box } from '@chakra-ui/react'

const GalleryPost = ({
  data: { previous, next, site, markdownRemark: post, allFile },
  location,
}) => {
  const siteTitle = site.siteMetadata?.title || 'Title'

  const images = allFile.nodes
    .filter(
      (image) =>
        image.relativeDirectory === post.fields.name &&
        image.childImageSharp &&
        !image.name.includes('icon'),
    )
    .map((image) => image.childImageSharp)

  return (
    <Layout location={location} title={siteTitle}>
      <header>
        <h1 style={{ margin: 0 }} itemProp="headline">
          {post.frontmatter.title}
        </h1>
        <p>{post.frontmatter.date}</p>
      </header>

      <section
        dangerouslySetInnerHTML={{ __html: post.html }}
        itemProp="articleBody"
      />
      <Box paddingBottom={4}>
        <Gallery images={images} mdColWidth={50} />
      </Box>
      <hr />
      <footer>
        <Bio />
      </footer>
      <nav className="blog-post-nav">
        <ul
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            justifyContent: 'space-between',
            listStyle: 'none',
            padding: 0,
          }}
        >
          <li>
            {!previous && (
              <li>
                <Link to={'/'} rel="prev">
                  ← Home page
                </Link>
              </li>
            )}
            {previous && (
              <Link to={previous.fields.slug} rel="prev">
                ← {previous.frontmatter.title}
              </Link>
            )}
          </li>
          <li>
            {next && (
              <Link to={next.fields.slug} rel="next">
                {next.frontmatter.title} →
              </Link>
            )}
            {!next && (
              <li>
                <Link to={'/'} rel="prev">
                  Home page →
                </Link>
              </li>
            )}
          </li>
        </ul>
      </nav>
    </Layout>
  )
}

export const Head = ({ data: { markdownRemark: post } }) => {
  return (
    <Seo
      title={post.frontmatter.title}
      description={post.frontmatter.description || post.excerpt}
    />
  )
}

export default GalleryPost

export const pageQuery = graphql`
  query GalleryPostBySlug(
    $id: String!
    $previousPostId: String
    $nextPostId: String
  ) {
    site {
      siteMetadata {
        title
      }
    }

    allFile {
      nodes {
        childImageSharp {
          thumb: gatsbyImageData(width: 400, height: 400, placeholder: BLURRED)
          full: gatsbyImageData(layout: FULL_WIDTH)
        }
        name
        relativeDirectory
      }
    }

    markdownRemark(id: { eq: $id }) {
      id
      excerpt(pruneLength: 160)
      html
      fields {
        slug
        name
      }
      frontmatter {
        title
        date(formatString: "MMMM DD, YYYY")
        description
      }
    }

    previous: markdownRemark(id: { eq: $previousPostId }) {
      fields {
        slug
      }
      frontmatter {
        title
      }
    }

    next: markdownRemark(id: { eq: $nextPostId }) {
      fields {
        slug
      }
      frontmatter {
        title
      }
    }
  }
`
