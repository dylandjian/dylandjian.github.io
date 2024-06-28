import React from 'react'
import { Link } from 'gatsby'
import {
  Heading,
  Text,
  VStack,
  HStack,
  Box,
  useMediaQuery,
  Stack,
} from '@chakra-ui/react'
import { GatsbyImage, getImage } from 'gatsby-plugin-image'

const BlogPostText = ({ post }) => {
  const [isLargerThan500] = useMediaQuery('(min-width: 500px)')

  if (isLargerThan500) {
    return (
      <Stack spacing={0}>
        <Heading fontSize={26} marginY={0}>
          {post.frontmatter.title}
        </Heading>
        <Text fontSize={15} as="i" marginBottom={2}>
          {`${post.frontmatter.date} - ${post.fields.readingTime.text}`}
        </Text>

        <Text fontSize={15} marginBottom={0}>
          {post.frontmatter.description || post.excerpt}
        </Text>
      </Stack>
    )
  }

  return (
    <Stack spacing={0}>
      <Heading fontSize={26} marginY={0}>
        {post.frontmatter.title}
      </Heading>
      <Text fontSize={15} marginBottom={0}>
        {post.frontmatter.date}
      </Text>
      <Text fontSize={15} as="i" marginBottom={0}>
        {post.fields.readingTime.text}
      </Text>
    </Stack>
  )
}

export const BlogPosts = ({ posts }) => {
  if (posts.length === 0) {
    return <Text>No blog posts found.</Text>
  }

  return (
    <VStack spacing={8} paddingTop={4} alignItems="flex-start">
      {posts.map((post: any) => {
        return (
          <Link
            to={post.fields.slug}
            style={{ textDecoration: 'none', color: 'inherit' }}
            key={post.fields.name}
          >
            <HStack spacing={4}>
              <Box minWidth={120} height={120}>
                <GatsbyImage
                  objectFit="cover"
                  image={getImage(post.icon.childImageSharp)}
                  alt="Caffe Latte"
                />
              </Box>

              <VStack spacing={0} align="flex-start">
                <BlogPostText post={post} />
              </VStack>
            </HStack>
          </Link>
        )
      })}
    </VStack>
  )
}
