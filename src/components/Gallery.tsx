import { Box, HStack, Heading, Text, VStack } from '@chakra-ui/react'
import { Link } from 'gatsby'
import { GatsbyImage, getImage } from 'gatsby-plugin-image'
import React from 'react'

export const Gallery = ({ images }) => {
  if (images.length === 0) {
    return <Text>No blog images found.</Text>
  }

  return (
    <VStack spacing={8} paddingTop={4}>
      {images.map((post: any) => {
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
