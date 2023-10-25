import * as React from 'react'
import { Text, Flex, VStack, HStack } from '@chakra-ui/react'
import { StaticImage } from 'gatsby-plugin-image'

export function Bio() {
  return (
    <Flex className="bio">
      <HStack>
        <Flex width={200}>
          <StaticImage
            className="bio-avatar"
            layout="fixed"
            formats={['auto', 'webp', 'avif']}
            src="../images/profile-pic.png"
            width={200}
            height={200}
            quality={100}
            alt="Profile picture"
          />
        </Flex>
        <VStack align="start">
          <Text fontSize="xl" as="b">
            Welcome to my blog !
          </Text>
          <Text>
            I am a Software Engineer, mostly self-taught and passionated about a
            lot of different topics.
          </Text>
          <Text>
            On this blog, you will find blog posts about anything, recipes,
            photos and any other endeavors I may find interesting to share.
          </Text>
          <Text> Happy browsing !</Text>
        </VStack>
      </HStack>
    </Flex>
  )
}
