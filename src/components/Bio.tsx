import * as React from 'react'
import { Text, Flex, Stack, VStack, HStack } from '@chakra-ui/react'
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
            width={100}
            height={100}
            quality={95}
            alt="Profile picture"
          />
        </Flex>
        <VStack align="start">
          <Text fontSize="xl" as="b">
            Welcome to my blog !
          </Text>
          <Text>
            I am a Software Engineer, current working for{' '}
            <a href="https://payfit.com">PayFit</a>, mostly self-taught and
            passionated about a lot of different topics.
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
