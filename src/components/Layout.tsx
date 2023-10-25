import React, { ReactNode } from 'react'
import { Link } from 'gatsby'
import { StaticImage } from 'gatsby-plugin-image'
import { Text, HStack } from '@chakra-ui/react'

function Contact() {
  return (
    <HStack spacing={4} marginLeft="auto">
      <Link to="https://github.com/dylandjian">
        <StaticImage
          layout="fixed"
          src="../../static/github.svg"
          alt="GitHub"
          width={28}
          height={28}
        />
      </Link>
      <Link to="https://twitter.com/dylandjian">
        <StaticImage
          layout="fixed"
          src="../../static/x.svg"
          alt="X"
          width={28}
          height={28}
        />
      </Link>
      <Link to="https://www.linkedin.com/in/dylan-djian">
        <StaticImage
          layout="fixed"
          src="../../static/linkedin.svg"
          alt="LinkedIn"
          width={28}
          height={28}
        />
      </Link>
    </HStack>
  )
}

function HomeHeader({ title }: { title: string }) {
  return (
    <h1 className="main-heading">
      <HStack spacing={42}>
        <Text margin={0}>{title}</Text>
        <Contact />
      </HStack>
    </h1>
  )
}

function PageHeader({ title }: { title: string }) {
  return (
    <Link className="header-link-home" to="/">
      <HStack spacing={42}>
        <Text margin={0}>{title}</Text>
      </HStack>
    </Link>
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
