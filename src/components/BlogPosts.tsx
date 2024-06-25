import React from 'react'
import { Link } from 'gatsby'
import { Heading, Text, VStack, HStack, Box } from '@chakra-ui/react'
import { GatsbyImage, getImage } from 'gatsby-plugin-image'

export const BlogPosts = ({ posts }) => {
  if (posts.length === 0) {
    return <Text>No blog posts found.</Text>
  }

  console.log(posts)

  return (
    <VStack spacing={8} paddingTop={4} alignItems="flex-start">
      {posts.map((post: any) => {
        return (
          <HStack spacing={4} key={post.fields.name}>
            <Box minWidth={120} height={120}>
              <GatsbyImage
                objectFit="cover"
                image={getImage(post.icon.childImageSharp)}
                alt="Caffe Latte"
              />
            </Box>

            <VStack spacing={0} align="flex-start">
              <Link to={post.fields.slug}>
                <Heading fontSize={26} marginY={0}>
                  {post.frontmatter.title}
                </Heading>
              </Link>
              <Text fontSize={14} marginBottom={2}>
                {post.frontmatter.date}
              </Text>
              <Text fontSize={15} marginBottom={0}>
                {post.frontmatter.description || post.excerpt}
              </Text>
            </VStack>
          </HStack>
        )
      })}
    </VStack>
  )
}
