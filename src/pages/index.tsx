import React from 'react'
import { graphql } from 'gatsby'
import { Tab, TabList, TabPanel, TabPanels, Tabs } from '@chakra-ui/react'

import { Bio } from '../components/Bio'
import { Layout } from '../components/Layout'
import { BlogPosts } from '../components/BlogPosts'

const BlogIndex = ({ data, location }) => {
  const siteTitle = data.site.siteMetadata?.title || 'Title'

  return (
    <Layout location={location} title={siteTitle}>
      <Bio />
      <Tabs isFitted defaultIndex={0}>
        <TabList>
          <Tab>Posts</Tab>
          <Tab>Recipe</Tab>
          <Tab>Portfolio</Tab>
        </TabList>
        <TabPanels>
          <TabPanel>
            <BlogPosts data={data} />
          </TabPanel>
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

    allMarkdownRemark(sort: { fields: [frontmatter___date], order: DESC }) {
      nodes {
        excerpt
        fields {
          slug
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
