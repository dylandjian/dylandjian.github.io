import React from 'react'
import { graphql } from 'gatsby'
import { Tab, TabList, TabPanel, TabPanels, Tabs } from '@chakra-ui/react'

import { Bio } from '../components/Bio'
import { Layout } from '../components/Layout'
import { BlogPosts } from '../components/BlogPosts'
import { Gallery } from '../components/Gallery'

const BlogIndex = ({ data, location }) => {
  const siteTitle = data.site.siteMetadata?.title || 'Title'

  const blogPosts = data.allMarkdownRemark.nodes.filter(
    (elem: any) => elem.fields.type === 'blog',
  )
  const enrichedBlogPosts = blogPosts.reduce((acc, elem) => {
    const name = elem.fields.name

    const icon = data.allFile.nodes.find(
      (node) =>
        node.childImageSharp &&
        node.childImageSharp.fixed.originalName === `${name}-icon.png`,
    )

    return [...acc, { ...elem, icon }]
  }, [])

  const galleryPosts = data.allMarkdownRemark.nodes.filter(
    (elem: any) => elem.fields.type === 'gallery',
  )
  const enrichedGalleryPosts = galleryPosts.reduce((acc, elem) => {
    const allFiles = data.allFile.nodes
    const name = elem.fields.name

    const icon = allFiles.find(
      (node) =>
        node.childImageSharp &&
        node.childImageSharp.fixed.originalName === `${name}-icon.png`,
    )
    const relatedImages = allFiles.filter(
      (file) => file.extension !== 'md' && file.relativeDirectory === name,
    )

    console

    return [...acc, { ...elem, images: relatedImages, icon }]
  }, [])

  return (
    <Layout location={location} title={siteTitle}>
      <Bio />
      <Tabs isFitted defaultIndex={0}>
        <TabList>
          <Tab>Posts</Tab>
          <Tab>Gallery</Tab>
          <Tab>Recipes</Tab>
        </TabList>
        <TabPanels>
          <TabPanel>
            <BlogPosts posts={enrichedBlogPosts} />
          </TabPanel>
          <TabPanel>
            <Gallery images={enrichedGalleryPosts} />
          </TabPanel>
          <TabPanel>No recipes yet</TabPanel>
        </TabPanels>
      </Tabs>
    </Layout>
  )
}

export default BlogIndex

export const pageQuery = graphql`
  query {
    site {
      siteMetadata {
        title
      }
    }

    allFile {
      nodes {
        childImageSharp {
          id
          gatsbyImageData(
            width: 120
            height: 120
            placeholder: BLURRED
            formats: [AUTO, WEBP, AVIF]
          )
          fixed {
            originalName
          }
        }
        id
        relativeDirectory
        name
        extension
        absolutePath
      }
    }

    allMarkdownRemark(sort: { frontmatter: { date: DESC } }) {
      nodes {
        excerpt
        fields {
          type
          slug
          name
        }
        frontmatter {
          date(formatString: "MMMM DD, YYYY")
          title
          description
        }
      }
    }
  }
`
