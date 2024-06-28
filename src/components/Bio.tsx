import * as React from 'react'
import { Text, Flex, VStack, useMediaQuery, Stack } from '@chakra-ui/react'
import { StaticImage } from 'gatsby-plugin-image'

export function Bio() {
  const [isLargerThan500] = useMediaQuery('(min-width: 500px)')

  return (
    <Flex className="bio" marginBottom="12px">
      <Stack
        direction={isLargerThan500 ? 'row' : 'column'}
        alignItems="center"
        spacing={8}
      >
        <Flex>
          <StaticImage
            layout="fixed"
            formats={['auto', 'webp', 'avif']}
            src="../images/profile-pic.png"
            width={200}
            height={200}
            style={{
              borderRadius: '20px',
              minWidth: '50px',
            }}
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
      </Stack>
    </Flex>
  )
}
