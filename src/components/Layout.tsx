import React, { ReactNode } from 'react'
import { Link } from 'gatsby'
import { StaticImage } from 'gatsby-plugin-image'
import { Text, HStack, useMediaQuery } from '@chakra-ui/react'

function Contact() {
  const [isLargerThan500] = useMediaQuery('(min-width: 500px)')

  if (isLargerThan500) {
    return (
      <HStack spacing={4} marginLeft="auto">
        <a href="https://github.com/dylandjian">
          <StaticImage
            layout="fixed"
            src="../../static/github.svg"
            alt="GitHub"
            width={28}
            height={28}
          />
        </a>
        <a href="https://twitter.com/dylandjian">
          <StaticImage
            layout="fixed"
            src="../../static/x.svg"
            alt="X"
            width={28}
            height={28}
          />
        </a>
        <a href="https://www.linkedin.com/in/dylan-djian">
          <StaticImage
            layout="fixed"
            src="../../static/linkedin.svg"
            alt="LinkedIn"
            width={28}
            height={28}
          />
        </a>
      </HStack>
    )
  }

  return (
    <HStack spacing={4} marginLeft="auto">
      <a href="https://github.com/dylandjian">
        <StaticImage
          layout="fixed"
          src="../../static/github.svg"
          alt="GitHub"
          width={22}
          height={22}
        />
      </a>
      <a href="https://twitter.com/dylandjian">
        <StaticImage
          layout="fixed"
          src="../../static/x.svg"
          alt="X"
          width={22}
          height={22}
        />
      </a>
      <a href="https://www.linkedin.com/in/dylan-djian">
        <StaticImage
          layout="fixed"
          src="../../static/linkedin.svg"
          alt="LinkedIn"
          width={22}
          height={22}
        />
      </a>
    </HStack>
  )
}

function HomeHeader({ title }: { title: string }) {
  const [isLargerThan500] = useMediaQuery('(min-width: 500px)')

  return (
    <h1 className="main-heading">
      <HStack>
        <HStack>
          <Text margin={0} fontSize={isLargerThan500 ? '4xl' : '3xl'}>
            {title}
          </Text>
        </HStack>
        <Contact />
      </HStack>
    </h1>
  )
}

function PageHeader({ title }: { title: string }) {
  const [isLargerThan500] = useMediaQuery('(min-width: 500px)')

  return (
    <HStack>
      <Link className="header-link-home" to="/">
        <HStack>
          <StaticImage
            className="bio-avatar"
            layout="fixed"
            formats={['auto', 'webp', 'avif']}
            src="../images/profile-pic.png"
            width={75}
            height={75}
            quality={100}
            alt="Profile picture"
          />
          <Text margin={0} fontSize={isLargerThan500 ? '4xl' : 'xl'}>
            {title}
          </Text>
        </HStack>
      </Link>
      <Contact />
    </HStack>
  )
}

export function Layout({
  location,
  title,
  children,
}: {
  location: string
  title: string
  children: ReactNode
}) {
  // @ts-expect-error because PATH PREFIX not defined
  const isRootPath = location.pathname === `${__PATH_PREFIX__}/`

  return (
    <div className="global-wrapper" data-is-root-path={isRootPath}>
      <header className="global-header">
        {isRootPath ? (
          <HomeHeader title={title} />
        ) : (
          <PageHeader title={title} />
        )}
      </header>
      <main>{children}</main>
    </div>
  )
}
