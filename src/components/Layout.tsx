import React, { ReactNode } from 'react'
import { Link } from 'gatsby'

function HomeHeader({ title }: { title: string }) {
  return (
    <h1 className="main-heading">
      <Link to="/">{title}</Link>
    </h1>
  )
}

function PageHeader({ title }: { title: string }) {
  return (
    <Link className="header-link-home" to="/">
      {title}
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
